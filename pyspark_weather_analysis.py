from pyspark import SparkContext, SparkFiles
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession, SQLContext, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from functools import reduce
import timeit
def main():
    """
    main function to parse, clean up and analyze data
    :return:
    """
    # create empty list to append data and row count for each station
    df_all = []
    station_count = []
    station_list = ["aberporth",
                    "armagh",
                    "ballypatrick",
                    "bradford",
                    "braemar",
                    "camborne",
                    "cambridge",
                    "cardiff",
                    "chivenor",
                    "dunstaffnage",
                    "durham",
                    "eastbourne",
                    "eskdalemuir",
                    "heathrow",
                    "hurn",
                    "lerwick",
                    "leuchars",
                    "whitby",
                    "cwmystwyth",
                    "lowestoft",
                    "manston",
                    "nairn",
                    "newtonrigg",
                    "oxford",
                    "paisley",
                    "ringway",
                    "rossonwye",
                    "shawbury",
                    "sheffield",
                    "southampton",
                    "stornoway",
                    "suttonbonington",
                    "tiree",
                    "valley",
                    "waddington",
                    "wickairport",
                    "yeovilton"]
    start_time = timeit.default_timer()

    # read data for each station and append to list
    for station in station_list:
        url = f"https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/{station}data.txt"
        sc.addFile(url)
        rdd = sc.textFile("file://" + SparkFiles.get(f"{station}data.txt"))
        df = read_data(rdd, station)
        station_count.append(df.count())
        df_all.append(df)

    # create dict for row count of each station
    station_count = dict(zip(station_list, station_count))
    print("Row count for each station:", station_count)

    # union dict of station data into complete df as all history data
    df_complete = reduce(DataFrame.unionAll, df_all).cache()

    # remove cached df
    del df_complete

    # finish timing program
    end_time = timeit.default_timer()
    print(f"Program duration: {end_time - start_time} seconds")


def read_data(rdd, station):
    """
    function to remove header line, special character and provisional data
    :param rdd:
    :param station:
    :return:
    """
    # list of df column name
    new_colnames = ["station", "year", "month", "tmax", "tmin", "af", "rain", "sun"]

    # finc the index for the end of header and filter out header & special character
    head_index = rdd.zipWithIndex().lookup("              degC    degC    days      mm   hours")
    cli_data = rdd.zipWithIndex() \
        .filter(lambda row_index: row_index[1] > head_index[0]).keys() \
        .filter(lambda x: any(e not in x for e in ["*", "#", "$"]))

    # split rdd, add station name, convert to df
    line_df = cli_data.map(lambda line: (station, line.split(" "))).toDF(("station", "data"))

    # remove empty array from data column
    # for spark >= 2.4 use array_remove function
    # from pyspark.sql.function import array_remove
    # line_split.withColumn("data", array_remove("data", ""))
    drop_array = udf(drop_from_array, ArrayType(StringType()))

    # remove rows with extra character for provisional data
    df = line_df.withColumn("data", drop_array("data", lit(""))).filter(size("data") == 7)

    # split column data to multiple columns and rename new column
    df1 = df.select([df.station] + [df.data[i] for i in range(7)])
    df2 = df1.toDF(*new_colnames) \
        .replace("___", None) \
        .na.fill({"tmax": 0.0, "tmin": 0.0, "af": 0.0, "rain": 0.0, "sun": 0.0}) \
        .withColumn("year", col("year").cast("integer")) \
        .withColumn("month", col("month").cast("integer")) \
        .withColumn("tmax", col("tmax").cast("float")) \
        .withColumn("tmin", col("tmin").cast("float")) \
        .withColumn("af", col("af").cast("float")) \
        .withColumn("rain", col("rain").cast("float")) \
        .withColumn("sun", col("sun").cast("float")) \
        .cache()
    return df2


def drop_from_array(arr, item):
    """
    function to remove item from array
    """
    return [x for x in arr if x != item]

if __name__ == "__main__":
    # define Spark configration
    conf = SparkConf().setAppName("Spark Transformation App")
    conf.set("spark.executor.memoryOverhead", "4096")
    conf.set("spark.driver.memoryOverhead", "4096")
    conf.set("spark.dynamicAllocation.executorIdleTimeout", "60")
    conf.set("spark.dynamicAllocation.minExecutors", "2")
    conf.set("spark.dynamicAllocation.initialExecutors", "2")
    conf.set("spark.sql.shuffle.partitions", "200")
    conf.set("spark.sql.autoBroadcastJoinThreshold", 1024 * 1024 * 10)
    conf.set("spark.yarn.maxAppAttempts", 1)
    conf.set("spark.yarn.max.executor.failures", 100)
    conf.set("spark.sql.broadcastTimeout", 36000)

    # define Spark driver session
    spark = SparkSession \
        .builder.master("local[*]") \
        .config(conf=conf) \
        .appName("ClimateAnalysis") \
        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    print("END")
    main()