from pyspark import SparkConf, SparkContext,SQLContext
# SparkContext.setSystemProperty("hadoop.home.dir", "C:\\bigdata\\spark-1.5.1-bin-hadoop2.6\\")
import sys, operator
import json
import string
import re
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import nltk
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
import pickle

# path to the nltk data directory.
# nltk.data.path.append("C:\\Users\\Dell\\Desktop\\bd-inputs\\nltk_data")
nltk.data.path.append("/cs/vml2/avahdat/CMPT733_Data_Sets/Assignment3/nltk_data")

clean_list=[]
dictionary={'1':3}
word_vectors=[]
final_list=[]

def clean_words(line):
 s = re.sub(r'[^\w\s]',' ',line)
 fin_s=s.lower()
 clean_list=re.sub(' +', ' ',fin_s).strip().split(' ')
 return clean_list


inputs = sys.argv[1]

conf = SparkConf().setAppName('word-vec')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

schema = StructType([
    StructField('reviewText', StringType(), False)
])

df = sqlContext.read.json(inputs, schema=schema)
df.registerTempTable('review_table')
sd=sqlContext.sql("""
    SELECT reviewText FROM review_table
""")
fin=sd.rdd.map(lambda x: str(x.reviewText)).map(clean_words).cache()

word2vec = Word2Vec()
model = word2vec.fit(fin)

# with open('model.pickle', 'wb') as g:
# 	pickle.dump(model, g)

result=fin.flatMap(lambda line:line).distinct().cache()

unique_list=result.collect()

for i in unique_list:
    try:
     vector=model.transform(i)
     word_vectors.append(vector)
     dictionary[i]=vector

    except Exception,e:
      pass

with open('vectorDictionary.pickle', 'wb') as f:
	pickle.dump(dictionary, f)

vectors_rdd=sc.parallelize(word_vectors)
num_clusters=2000

# Build the model (cluster the data)
clusters = KMeans.train(vectors_rdd, num_clusters, maxIterations=10,
        runs=10, initializationMode="random")

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = vectors_rdd.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

final_rdd=sc.parallelize(dictionary.keys())
final_r={}
for i in dictionary.keys():
    try:
     final_r[i] = (clusters.predict(dictionary[i]))

    except Exception,e:
     pass

with open('clusterFinal.pickle', 'wb') as f:
	pickle.dump(final_r, f)
