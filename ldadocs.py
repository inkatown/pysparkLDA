from __future__ import print_function
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.mllib.linalg import SparseVector
from collections import Counter
import sys
from operator import add
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.types import IntegerType
spark = SparkSession\
        .builder\
        .appName("bagofwords")\
        .getOrCreate()
sc = spark.sparkContext
datRdd = sc.wholeTextFiles("/mydata/temp2",4)
df  = datRdd.toDF(["path","doc"])
regexTokenizer = RegexTokenizer(inputCol="doc", outputCol="words", pattern="\\W")
regexTokenized = regexTokenizer.transform(df)
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
df_af_rem = remover.transform(regexTokenized)
NoStopRdd = (df_af_rem.select("filtered")).rdd.flatMap(list)
local_vocab_map = NoStopRdd \
        .flatMap(lambda token: token) \
        .distinct() \
        .zipWithIndex() \
        .collectAsMap()
vocab_map = sc.broadcast(local_vocab_map)
vocab_size = sc.broadcast(len(local_vocab_map))
term_document_matix = NoStopRdd \
      .map(Counter) \
      .map(lambda counts: {vocab_map.value[token]: float(counts[token]) for token in local_vocab_map})
parsedData = term_document_matix.map(lambda line: Vectors.dense([line[x] for x in line.keys()]))
corpus = parsedData.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()
# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=5)
topics = ldaModel.describeTopics()
for topic in range(5):
    print("Topic " + str(topic) + ":")
    for word in range(0, 20):
        print(list(local_vocab_map.keys())[list(local_vocab_map.values()).index((topics[topic][0])[word])]+" " +
 str(topic)+" "+ str((topics[topic][1])[word]))
