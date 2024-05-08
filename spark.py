from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Predict Phone Cost").getOrCreate()

phone_cell_train_df = spark.read.format("csv").option("sep", ",").load("CellPhone_train.csv", header=True)
phone_cell_test_df = spark.read.format("csv").option("sep", ",").load("CellPhone_test.csv", header=True)


# phone_cell_train_df.show()
# phone_cell_test_df.show()

phone_cell_train_df.print
# phone_cell_train_df.info()
