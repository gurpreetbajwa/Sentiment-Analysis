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
from pyspark.mllib.linalg import Vectors

# path to the nltk data directory.
# nltk.data.path.append("C:\\Users\\Dell\\Desktop\\bd-inputs\\nltk_data")
nltk.data.path.append("/cs/vml2/avahdat/CMPT733_Data_Sets/Assignment3/nltk_data")

clean_list = []
word_vectors = []
final_list = []

def clean_words(line):
 s = re.sub(r'[^\w\s]',' ',line)
 fin_s=s.lower()
 clean_list=re.sub(' +', ' ',fin_s).strip().split(' ')
 return clean_list

def get_sparseVector(x):
    ids=[]
    for j in x:
        if j in cluster.keys():
            ids.append(cluster[j])

    bag_words = {}
    for i in ids:
        bag_words[i]=(float(ids.count(i))/len(ids))
     # Create a SparseVector
    sv = Vectors.sparse(2000, bag_words)
    return sv


input = sys.argv[1]
output = sys.argv[2]

conf = SparkConf().setAppName('bag_words')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

with open('clusterFinal.pickle', 'rb') as f:
	cluster=pickle.load(f)


schema = StructType([
    StructField('reviewText', StringType(), False),StructField('overall', FloatType(), False),StructField('reviewTime', StringType(), False)
])

df = sqlContext.read.json(input, schema=schema)
df.registerTempTable('review_table')
sd=sqlContext.sql("""
    SELECT reviewText FROM review_table
""")
fin=sd.rdd.map(lambda x: str(x.reviewText)).map(clean_words)
sparse_vectors=fin.map(get_sparseVector)
time=sqlContext.sql("""
    SELECT reviewTime FROM review_table
""")
time_split=time.rdd.map(lambda x: str(x.reviewTime)).map(lambda line: line.split(', '))
year_list=time_split.map(lambda (x,y):y).collect()

score=sqlContext.sql("""
    SELECT overall FROM review_table
""")
score_list=score.rdd.map(lambda x:str(x.overall)).collect()
sparse_list=sparse_vectors.collect()
zip_list=zip(sparse_list, year_list, score_list)
zip_rdd=sc.parallelize(zip_list)

zip_train=zip_rdd.filter(lambda (x,y,z): y!= '2014').map(lambda (x,y,z):(x,z)).coalesce(1)
zip_test=zip_rdd.filter(lambda (x,y,z): y == '2014').map(lambda (x,y,z):(x,z)).coalesce(1)

# zip_train.saveAsPickleFile(output+"/bow_train")
# zip_test.saveAsPickleFile(output+"/bow_test")
zip_train.saveAsPickleFile(output+"/bow_train_small")
zip_test.saveAsPickleFile(output+"/bow_test_small")
