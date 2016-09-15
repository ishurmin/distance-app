package com.challenge.distance;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import static org.apache.spark.mllib.random.RandomRDDs.uniformJavaVectorRDD;

/**
 * Класс-задание для Apache Spark, который создает набор случайных векторов Vi и для заданного
 * случайного вектора X находит ближайший вектор Vk
 */
public class App
{
    private static String NAME = "Distance App";
    private static String OUTPUT_FILE = "output.txt";
    private static int NUMBER_OF_VECTORS = 1000000;
    private static int VECTOR_SIZE = 100;

    /**
     * Создает массив со случайными вещественными числами от 0 до 1
     * @param size - размер массива
     * @return double[]
     */
    private static double[] createRandomArray(final int size)
    {
        final Random rand = new Random();
        final double[] arr = new double[size];
        for (int i = 0; i < size; i++) {
            arr[i] = rand.nextDouble();
        }
        return arr;
    }

    public static void main( String[] args ) throws IOException {

        int numberOfVectors, vectorSize;
        try {
            numberOfVectors = Integer.parseInt(args[0]);
            vectorSize = Integer.parseInt(args[1]);
        } catch (final ArrayIndexOutOfBoundsException | NumberFormatException e) {
            numberOfVectors = NUMBER_OF_VECTORS;
            vectorSize = VECTOR_SIZE;
        }

        final SparkConf conf = new SparkConf().setAppName(NAME);
        final JavaSparkContext jsc = new JavaSparkContext(conf);

        // 1. далее мы создадим RDD и тут же его используем,
        //    но вообще можно загрузить данные из файла
        // 2. а еще можно использовать DataFrame и SQL :)

        // создаем RDD с парами (индекс, вектор)
        final JavaPairRDD<Long, Vector> rddVectors =
            // генерируем случайные вещественные вектора
            uniformJavaVectorRDD(jsc, numberOfVectors, vectorSize)
            // добавляем индексы
            .zipWithIndex()
            // меняем индексы и вектора в паре местами
            .mapToPair(
                pair -> new Tuple2<>(pair._2(), pair._1())
            // кешируем RDD, т.к. еще будем использовать для поиска нужного вектора
            ).cache();

        // создаем вектор X
        final Broadcast<Vector> x = jsc.broadcast(Vectors.dense(createRandomArray(vectorSize)));

        // создаем RDD с парами (индекс, расстояние)
        // для расчета расстояния используем Squared Euclidean distance
        final JavaPairRDD<Long, Double> rddDistances = rddVectors.mapToPair(
            vector -> new Tuple2<>(vector._1(), Vectors.sqdist(vector._2(), x.value()))
        );

        // находим минимальное расстояние, возращаем пару (индекс, расстояние)
        final Tuple2<Long, Double> minDist = rddDistances.min(Comparator.comparing(
            (Function<Tuple2<Long,Double>, Double> & Serializable) Tuple2::_2
        ));

        // находим вектор по индексу
        final Vector v = rddVectors.lookup(minDist._1()).iterator().next();

        final List<String> output = new ArrayList<>();
        output.add("number of vectors: " + numberOfVectors);
        output.add("vector size: " + vectorSize);
        output.add("minimum distance: " + minDist._2());
        output.add("nearest neighbour - vector #" + minDist._1() + ":");
        output.add(v.toString());
        output.add("tested vector:");
        output.add(x.value().toString());
        Files.write(Paths.get(OUTPUT_FILE), output);
    }
}
