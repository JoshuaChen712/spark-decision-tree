from df_decision_tree import DFDecisionTreeModel, FeatureType
from pyspark.sql import SparkSession
import time

partition_num = 8

spark = SparkSession \
    .builder \
    .appName("Experiments") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

## Load dataset
df = spark.read.csv("test_dataset_3.csv", header=True, inferSchema=True)

## Split dataset into train data and test data
(train_data, test_data) = df.randomSplit([0.7, 0.3], seed=42)
train_data = train_data.repartition(partition_num)
## Explicitly declare the feature type (Could be be optimized for self-inference)
featureTypes = {"feature1": FeatureType.CONTINOUS, "feature2": FeatureType.DISCRETE, "feature3": FeatureType.DISCRETE}

## Define the dicision tree model.
model = DFDecisionTreeModel(featureTypes = featureTypes, treeType="C4.5", maxBins =5)

## Calculate time for training (building the decision tree)
start_time = time.time()
model.train(train_data)
end_time = time.time()

duration = end_time - start_time
print("Execution Time{:.2f}s".format(duration))

## Evaluate the performance
model.test(test_data)