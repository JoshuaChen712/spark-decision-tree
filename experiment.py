from df_decision_tree import DFDecisionTreeModel, FeatureType
from pyspark.sql import SparkSession
import time

partition_num = 10

spark = SparkSession \
    .builder \
    .appName("Experiments") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df = spark.read.csv("test_dataset.csv", header=True).repartition(partition_num)
(train_data, test_data) = df.randomSplit([0.7, 0.3], seed=42)
featureTypes = {"feature1": FeatureType.DISCRETE, "feature2": FeatureType.DISCRETE, "feature3": FeatureType.DISCRETE}
model = DFDecisionTreeModel(featureTypes = featureTypes, treeType="CART")

start_time = time.time()
model.train(train_data)
end_time = time.time()

duration = end_time - start_time
print("Execution Time{:.2f}s".format(duration))

model.test(test_data)