from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

conf = SparkConf().setAppName("DecisionTreeExample")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("test_dataset_2.csv")

assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
data = assembler.transform(data)
data = data.select("features", "target")

(trainingData, testData) = data.randomSplit([0.7, 0.3], seed=42)


dt = DecisionTreeClassifier(labelCol="target", featuresCol="features")
model = dt.fit(trainingData)
predictions = model.transform(testData)

evaluator = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(accuracy)

print(model.toDebugString)