from pyspark.sql import SparkSession
from pyspark.sql.functions import concat, lit, regexp_replace, regexp_extract, col
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder.appName("Predict Phone Cost").getOrCreate()
spark.conf.set('spark.sql.repl.eagerEval.enabled', True)
spark.conf.set('spark.sql.repl.eagerEval.maxColumnWidth', 1000)
spark.conf.set("spark.sql.repl.eagerEval.truncate", 1000)

dataset_1 = spark.read.format("csv").option("sep", ",").load("datasets/Dataset_1.csv", header=True)
dataset_2 = spark.read.format("csv").option("sep", ",").load("datasets/Dataset_2.csv", header=True)
dataset_3 = spark.read.format("csv").option("sep", ",").load("datasets/Dataset_3.csv", header=True)
dataset_4 = spark.read.format("csv").option("sep", ",").load("datasets/Dataset_4.csv", header=True)

# Удаляем столбцы, которые не будем использовать
dataset_1 = dataset_1\
    .withColumn("Phone Name", concat(dataset_1["Brand"], lit(" "), dataset_1["Model"]))\
    .drop("Brand", "Model", "Screen Size")
columns = dataset_1.columns
columns.remove("Phone Name")
columns.insert(0, "Phone Name")
dataset_1 = dataset_1.select(columns)\

# Приводим столбцы к нужному нам виду
dataset_1 = dataset_1\
    .withColumn("Storage", regexp_replace(dataset_1["Storage"], " GB", ""))\
    .withColumn("RAM", regexp_replace(dataset_1["RAM"], " GB", ""))\
    .withColumn("Camera", regexp_extract(dataset_1["Camera"], r"\d+", 0))\
    
# Приведём столбцы к нужному типу
dataset_1 = dataset_1\
    .withColumn("Storage", dataset_1["Storage"].cast("integer"))\
    .withColumn("RAM", dataset_1["RAM"].cast("integer"))\
    .withColumn("Camera", dataset_1["Camera"].cast("integer"))\
    .withColumn("Battery Capacity", dataset_1["Battery Capacity"].cast("integer"))\
    .withColumn("Price", dataset_1["Price"].cast("integer"))\
    .withColumn("Price", col("Price") * 80)\
    .withColumnRenamed("Battery Capacity", "Battery")

# Удаляем лишние столбцы и сокращаем названия
dataset_2 = dataset_2\
    .withColumnRenamed("Brand me", "Phone Name")\
    .withColumnRenamed("ROM", "Storage")\
    .withColumnRenamed("Primary_Cam", "Camera")\
    .withColumnRenamed("Battery_Power", "Battery")\
    .drop("Ratings", "Unnamed: 0", "Mobile_Size", "Selfi_Cam")

# Переопределяем типы у столбцов
dataset_2 = dataset_2\
    .withColumn("RAM", dataset_2["RAM"].cast("integer"))\
    .withColumn("Storage", dataset_2["Storage"].cast("integer"))\
    .withColumn("Camera", dataset_2["Camera"].cast("integer"))\
    .withColumn("Battery", dataset_2["Battery"].cast("integer"))\
    .withColumn("Price", dataset_2["Price"].cast("integer"))

# Создаём удобный порядок столбцов
columns_2 = dataset_2.columns
columns_2.remove("RAM")
columns_2.remove("Storage")
columns_2.insert(1, "Storage")
columns_2.insert(2, "RAM")
dataset_2 = dataset_2.select(columns_2)

# Удаляем лишние столбцы
dataset_3 = dataset_3\
    .drop("Rating ?/5", "Number of Ratings", "Processor", "Date of Scraping", "Front Camera")\
    .withColumnRenamed("ROM/Storage", "Storage")\
    .withColumnRenamed("Back/Rare Camera", "Camera")\
    .withColumnRenamed("Back/Rare Camera", "Battery")\
    .withColumnRenamed("Price in INR", "Price")

# Удаляем из строк лишние данные
dataset_3 = dataset_3\
    .withColumn("RAM", regexp_replace(dataset_3["RAM"], " GB RAM", ""))\
    .withColumn("Storage", regexp_replace(dataset_3["Storage"], " GB ROM", ""))\
    .withColumn("Battery", regexp_replace(dataset_3["Battery"], " mAh", ""))\
    .withColumn("Price", regexp_replace(regexp_replace(dataset_3["Price"], "₹", ""), ",", ""))\
    .withColumn("Camera", regexp_extract(dataset_3["Camera"], r"\d+", 0))\

# Меняем местами столбцы для нужного нам порядка
columns_3 = dataset_3.columns
columns_3.remove("RAM")
columns_3.remove("Storage")
columns_3.insert(1, "Storage")
columns_3.insert(2, "RAM")
dataset_3 = dataset_3.select(columns_3)

# Переопределяем типы у столбцов
dataset_3 = dataset_3\
    .withColumn("Storage", dataset_3["Storage"].cast("integer"))\
    .withColumn("RAM", dataset_3["RAM"].cast("integer"))\
    .withColumn("Camera", dataset_3["Camera"].cast("integer"))\
    .withColumn("Battery", dataset_3["Battery"].cast("integer"))\
    .withColumn("Price", (dataset_3["Price"] * 1.1).cast("integer"))\

columns_4 = ["Name", "Internal storage (GB)", "RAM (MB)", "Rear camera", "Battery capacity (mAh)","Price"]
dataset_4 = dataset_4.select(columns_4)
dataset_4 = dataset_4.withColumnRenamed("Battery capacity (mAh)", "Battery")\
    .withColumnRenamed("RAM (MB)", "RAM")\
    .withColumnRenamed("Internal Storage (GB)", "Storage")\
    .withColumnRenamed("Rear camera", "Camera")\
    .withColumnRenamed("Name", "Phone Name")

dataset_4 = dataset_4.withColumn("RAM", (dataset_4["RAM"] / 1000).cast("integer"))\
    .withColumn("Storage", dataset_4["Storage"].cast("integer"))\
    .withColumn("Camera", dataset_4["Camera"].cast("integer"))\
    .withColumn("Battery", dataset_4["Battery"].cast("integer"))\
    .withColumn("Price", dataset_4["Price"].cast("integer"))

# Объединим теперь наши датасеты в один
dataset_union = dataset_1.union(dataset_2).union(dataset_3).union(dataset_4).dropDuplicates(["Phone Name"]).na.drop()

# Создадим вектор фич для обучения
features = [x for x in dataset_union.columns if x != "Phone Name"]
print(features)
assembler = VectorAssembler(inputCols=features, outputCol="features")
feature_vector = assembler.transform(dataset_union)
# feature_vector

# Построим матрицу корреляций
# matrix = Correlation.corr(feature_vector, "features").collect()[0][0].toArray()
# sns.heatmap(matrix, xticklabels=features, yticklabels=features, cmap="Greens", annot=True)
# plt.show()

# Разделим датасет на два: тренировочный и тестовый.
train_df, test_df = feature_vector.randomSplit([0.8, 0.2], seed=42)

from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(labelCol="Price", featuresCol="features")
lr_model = lr.fit(train_df)
lr_predict = lr_model.transform(test_df)
lr_predict.show()