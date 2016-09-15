#### Описание

Программа-задание для **Apache Spark**, которая создает набор случайных векторов _Vi_ и для заданного случайного вектора _X_ находит ближайший вектор _Vk_.

В качестве расчета расстояния между векторами используется Squared Euclidean distance. В программе реализован простой перебор для подсчета расстояний, но даже для 1 млн векторов это занимает всего несколько секунд :) В случае низкой производительности могут быть применены k-d tree или R-tree.

Здесь нет тестов, но в продакшен-коде они, конечно же, должны быть ;) Для Spark набор инструментов пока что ограничен, см. [эту статью](http://www.jesse-anderson.com/2016/04/unit-testing-spark-with-java/).

Альтернативой этой реализации может быть использование  [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface), как на C/C++, так и на Java, через [биндинги для нативной библиотеки](http://www.open-mpi.de/faq/?category=java) или c помощью [собственной имплементации](http://mpj-express.org/). Краткий план подобной программы: на всех процессорах параллельно сгенерировать части общего набора векторов (генератор случ. чисел - [SPRNG](http://www.sprng.org/)), на мастер-процессоре сгенерировать тестируемый вектор, [MPI_Bcast](http://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/) его, на каждом процессоре параллельно подсчитать расстояния, найти минимальное и [MPI_Gather](http://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/) для окончательного нахождения минимума на мастер-процессоре.

#### Запуск

* Скачать и распаковать [Apache Spark](http://spark.apache.org/downloads.html).
* Выполнить `ПУТЬ_ДО_SPARK/bin/spark-submit --class=com.challenge.distance.App --master=local[4] --driver-memory 3G ПУТЬ_ДО_JAR/distance-app-1.0-SNAPSHOT.jar КОЛ-ВО_ВЕКТОРОВ РАЗМЕРНОСТЬ`
* Вывод будет записан в файл `output.txt` в текущем каталоге

JAR-файл находится в каталоге `target` проекта.

#### Параметры JAR
* `КОЛ-ВО_ВЕКТОРОВ` - необязательный, по-умолчанию 1000000
* `РАЗМЕРНОСТЬ` - необязательный, по-умолчанию 100

#### Параметры `spark-submit`
* `--class` - класс для запуска
* `--master` - [master-URL кластера Spark](http://spark.apache.org/docs/latest/submitting-applications.html#master-urls), `local[4]` - локальная тестовая версия, использующая четыре ядра
* `--driver-memory` - выделенная память, заданных 3G достаточно для кеширования 1 млн векторов

#### Сборка

* (если вдруг понадобится) Выполнить `mvn package` в корне проекта
