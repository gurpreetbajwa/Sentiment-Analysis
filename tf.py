from pyspark import SparkConf, SparkContext,SQLContext
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\bigdata\\spark-1.5.1-bin-hadoop2.6\\")
import sys, operator
import json
import string
import re
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import nltk
from nltk.corpus import stopwords
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.mllib.feature import Normalizer


# path to the nltk data directory.
# nltk.data.path.append("C:\\Users\\Dell\\Desktop\\bd-inputs\\nltk_data")
nltk.data.path.append("/cs/vml2/avahdat/CMPT733_Data_Sets/Assignment3/nltk_data")
stopword=set(stopwords.words('english'))

def clean_words(line):
 s = re.sub(r'[^\w\s]',' ',line)
 fin_s=s.lower()
 clean_list=re.sub(' +', ' ',fin_s).strip().split(' ')
 final_words=[x for x in clean_list if x not in stopword]

 return final_words


inputs = sys.argv[1]
output = sys.argv[2]


conf = SparkConf().setAppName('tf-idf')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
schema = StructType([
    StructField('reviewText', StringType(), False),StructField('overall', FloatType(), False),StructField('reviewTime', StringType(), False)
])

df = sqlContext.read.json(inputs, schema=schema)
df.registerTempTable('review_table')
sd=sqlContext.sql("""
    SELECT reviewText FROM review_table
""")

fin=sd.rdd.map(lambda x: str(x.reviewText)).map(clean_words)

#creating tf-idf
hashingTF = HashingTF()
tf = hashingTF.transform(fin)
tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf)
tf_idf_list=tfidf.collect()
# #Normalization
# normalizer1= Normalizer()
# tf_idf_list=normalizer1.transform(tfidf).collect()

time=sqlContext.sql("""
    SELECT reviewTime FROM review_table
""")

time_split=time.rdd.map(lambda x: str(x.reviewTime)).map(lambda line: line.split(', '))
year_list=time_split.map(lambda (x,y):y).collect()

score=sqlContext.sql("""
    SELECT overall FROM review_table
""")
score_list=score.rdd.map(lambda x:str(x.overall)).collect()

zip_list=zip(tf_idf_list, year_list, score_list)
zip_rdd=sc.parallelize(zip_list)

zip_train=zip_rdd.filter(lambda (x,y,z): y!= '2014').map(lambda (x,y,z):(x,z)).coalesce(1)
zip_test=zip_rdd.filter(lambda (x,y,z): y == '2014').map(lambda (x,y,z):(x,z)).coalesce(1)

zip_train.saveAsPickleFile(output+"/unnormalizedtrain")
zip_test.saveAsPickleFile(output+"/unnormalizedtest")
