from typing import Optional
from enum import Enum
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import log2, col
from pyspark.sql.types import IntegerType
from sklearn.metrics import accuracy_score
import queue


threshold = 0.005


class TreeNodeType(Enum):
    LEAF = 1
    INTERNAL = 2
    ROOT = 3


class FeatureType(Enum):
    CONTINOUS = 1
    DISCRETE = 2


class DecisionTreeNode():
    def __init__(self,
                 node_type: Optional[TreeNodeType] = None,
                 condition: Optional[str] = "",
                 split_feature: Optional[str] = None,
                 excluded_features: Optional[list[str]] = []):
        self.node_type = node_type
        self.condition = condition
        self.split_feature = split_feature
        self.excluded_features = excluded_features
        self.children = {}

    def updateSplitFeature(self, split_feature):
        self.split_feature = split_feature

    def change2Leaf(self,
                    prediction_class: int):
        self.node_type = TreeNodeType.LEAF
        self.prediction_class = prediction_class


class DecisionTree():
    DEBUG = False

    def __init__(self,
                 maxDepth: Optional[int] = 5,
                 maxBins: Optional[int] = 5,
                 numclasses: Optional[int] = None,
                 featureCols: Optional[list[str]] = None,
                 featureTypes: dict[str, FeatureType] = None,
                 labelCol: Optional[str] = None,
                 treeType: Optional[str] = "ID3"):
        self.maxDepth = maxDepth
        self.maxBins = maxBins
        self.numclasses = numclasses
        self.featureCols = featureCols
        self.featureTypes = featureTypes
        self.labelCol = labelCol
        self.treeType = treeType
        self.bins = {}
        self.splits_map = {}

    # binning and pre-split for all the features.
    def preprocessing(self,
                      data: DataFrame):
        self.data = data
        if self.featureCols == None:
            self.featureCols = self.data.columns[:-1]

        if self.labelCol == None:
            self.labelCol = self.data.columns[-1]
        for featureName, featureType in self.featureTypes.items():
            if (featureType == FeatureType.CONTINOUS):
                rdd = self.data.select(col(featureName)).rdd

                def find_min_max(iterator):
                    min_val = float('inf')
                    max_val = float('-inf')
                    for row in iterator:
                        val = row[0]
                        if val < min_val:
                            min_val = val
                        if val > max_val:
                            max_val = val
                    yield (min_val, max_val)

                min_max_rdd = rdd.mapPartitions(find_min_max)

                def reduce_min_max(a, b):
                    return (min(a[0], b[0]), max(a[1], b[1]))
                (min_val, max_val) = min_max_rdd.reduce(reduce_min_max)
                self.data = self.data.withColumn(featureName, ((col(
                    featureName) - min_val) / (max_val - min_val) * (self.maxBins - 1)).cast('int'))
                self.bins[featureName] = (min_val, max_val, self.maxBins)
                self.splits_map[featureName] = range(self.maxBins)
            else:
                featureValues = self.data.select(
                    featureName).distinct().collect()
                self.splits_map[featureName] = [row[featureName]
                                                for row in featureValues]

    # Use BFS to build the decision tree.
    def build_tree(self):
        node_queue = queue.Queue()
        self.root = DecisionTreeNode(node_type=TreeNodeType.ROOT)
        node_queue.put(self.root)
        while not node_queue.empty():
            node = node_queue.get()
            if node.node_type != TreeNodeType.ROOT:
                selected_columns = [
                    col for col in self.data.columns if col not in node.excluded_features]
                node_data = self.data.filter(
                    node.condition).select(selected_columns)
            else:
                node_data = self.data
            node_data.cache()
            if len(node_data.columns) == 1:
                self.change2Leaf(node, node_data)
                continue
            max_info_gain = 0
            max_info_gain_feature = None
            for featureName in node_data.columns:
                if featureName != self.labelCol:
                    info_gain = self.info_gain(node_data, featureName)
                    max_info_gain = max(max_info_gain, info_gain)
                    max_info_gain_feature = featureName if max_info_gain == info_gain else max_info_gain_feature
            if (max_info_gain < threshold):
                self.change2Leaf(node, node_data)
                continue
            splits = self.splits_map[max_info_gain_feature]
            node.updateSplitFeature(max_info_gain_feature)
            for split in splits:
                if node.condition != "":
                    new_condition = node.condition + " AND " + \
                        str(max_info_gain_feature) + " == " + str(split)
                else:
                    new_condition = str(
                        max_info_gain_feature) + "==" + str(split)
                new_excluded_features = node.excluded_features + \
                    [max_info_gain_feature]
                new_node = DecisionTreeNode(
                    node_type=TreeNodeType.INTERNAL, condition=new_condition, excluded_features=new_excluded_features)
                node_queue.put(new_node)
                node.children[split] = new_node

    # Mark the node as leaf, and using the majority of the node as the prediction class
    def change2Leaf(self,
                    node: DecisionTreeNode,
                    node_data: DataFrame):
        grouped_df = node_data.groupBy(self.labelCol).count()
        prediction_class = grouped_df.orderBy(
            "count", ascending=False).limit(1).collect()
        if len(prediction_class):
            node.change2Leaf(prediction_class[0][self.labelCol])

    # calculation for information gain. Different type tree using different methods
    def info_gain(self,
                  data: DataFrame,
                  featureName: str):
        total_num = float(data.count())
        if self.treeType == "ID3" or self.treeType == "C4.5":
            count_df = data.groupBy(self.labelCol).count()
            info_D = count_df.withColumn("info", -(count_df["count"])/total_num*log2(count_df["count"]/total_num))\
                .agg({"info": "sum"}).collect()[0][0]
            df1 = data.groupBy(featureName, self.labelCol).count()
            df2 = df1.groupBy(featureName).sum('count')
            joined_df = df1.join(df2, featureName)
            tmp_df = info = joined_df.withColumn('p', joined_df["count"]/joined_df['sum(count)'])\
                .withColumn('ratio', joined_df['sum(count)']/total_num)
            info = tmp_df.withColumn('info', tmp_df["ratio"]*(-tmp_df["p"]*log2(tmp_df["p"])))\
                .agg({"info": "sum"}).collect()[0][0]
            info_gain = info_D - info
            if self.treeType == "C4.5":
                tmp_df = joined_df.withColumn(
                    'ratio', joined_df['sum(count)']/total_num)
                split_info = tmp_df.withColumn('info', -tmp_df['ratio']*log2(tmp_df['ratio']))\
                    .agg({"info": "sum"}).collect()[0][0]
                if split_info:
                    info_gain = info_gain/split_info
                else:
                    info_gain = 1.0
            return info_gain
        if self.treeType == "CART":
            count_df = data.groupBy(self.labelCol).count()
            ratio_df = count_df.withColumn(
                "ratio", count_df["count"]/total_num)
            gini_D = ratio_df.withColumn("info", -ratio_df["ratio"]**2)\
                .agg({"info": "sum"}).collect()[0][0]
            df1 = data.groupBy(featureName, self.labelCol).count()
            df2 = df1.groupBy(featureName).sum('count')
            joined_df = df1.join(df2, featureName)
            tmp_df = joined_df.withColumn("p", joined_df["count"]/joined_df["sum(count)"])\
                .withColumn("ratio", joined_df["sum(count)"]/total_num)
            gini = tmp_df.withColumn("info", -tmp_df["ratio"]*tmp_df["p"]**2)\
                .agg({"info": "sum"}).collect()[0][0]
            info_gain = gini_D - gini
            return info_gain

    def predict(self, data_row):
        cur = self.root
        while (cur.node_type != TreeNodeType.LEAF):
            split_feature = cur.split_feature
            split = data_row[split_feature]
            cur = cur.children[split]
        return cur.prediction_class

    def evaluate(self,
                 test_data: DataFrame,
                 metrics: Optional[str] = "accuracy"):
        predictions = []
        for featureName, featureType in self.featureTypes.items():
            if featureType == FeatureType.CONTINOUS:
                (min_val, max_val, numBins) = self.bins[featureName]
                test_data = test_data.withColumn(featureName, ((
                    col(featureName) - min_val) / (max_val - min_val) * (numBins - 1)).cast('int'))
        featureCols = [
            col for col in test_data.columns if col != self.labelCol]
        features = test_data.select(featureCols).collect()
        for row in features:
            prediction = self.predict(row)
            predictions.append(prediction)
        labels = test_data.select(self.labelCol).collect()
        if (metrics == "accuracy"):
            score = accuracy_score(labels, predictions)
        return predictions, score

    def print_decision_tree(self):
        if DecisionTree.DEBUG:
            q = queue.Queue()
            q.put(self.root)
            while (not q.empty()):
                node = q.get()
                print("node_type", node.node_type)
                print("condition", node.condition)
                print("split_feature", node.split_feature)
                print("="*10)
                for key, child in node.children.items():
                    q.put(child)


class DFDecisionTreeModel():
    def __init__(self,
                 maxDepth: Optional[int] = 5,
                 maxBins: Optional[int] = 5,
                 numclasses: Optional[int] = None,
                 featureCols: Optional[list[str]] = None,
                 featureTypes: dict[str, FeatureType] = None,
                 labelCol: Optional[str] = None,
                 treeType: Optional[str] = "ID3"):
        self.decision_tree = DecisionTree(
            maxDepth, maxBins, numclasses, featureCols, featureTypes, labelCol, treeType)

    def train(self,
              train_data: DataFrame):
        self.decision_tree.preprocessing(train_data)
        self.decision_tree.build_tree()

    def test(self,
             test_data: DataFrame,
             metrics: Optional[str] = "accuracy"):
        self.decision_tree.print_decision_tree()
        (predictions, score) = self.decision_tree.evaluate(test_data, metrics)
        print(metrics, score)
