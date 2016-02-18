from pyspark import SparkConf, SparkContext
SparkContext.setSystemProperty("hadoop.home.dir", "C:\\spark-1.5.1-bin-hadoop2.6\\")
import sys, pickle,math
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils

conf = SparkConf().setAppName('random-forest')
sc = SparkContext(conf=conf)

input = sys.argv[1]

# Load and parse the data
def parsePoint(line):
    return LabeledPoint(float(line[1]), line[0])

train = sc.pickleFile(input+'/bow_train/part-00000')
test = sc.pickleFile(input+'/bow_test/part-00000')
parsedtrain=train.map(parsePoint).filter(lambda line:len(line.features)!=0 or len(line.label)!=0)
parsedtest = test.map(parsePoint).filter(lambda line:len(line.features)!=0 or len(line.label)!=0).cache()
model = GradientBoostedTrees.trainRegressor(parsedtrain,categoricalFeaturesInfo={}, numIterations=1)
predictions = model.predict(parsedtest.map(lambda x: x.features))
labelsAndPredictions = parsedtest.map(lambda lp: lp.label).zip(predictions)
val_err = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(parsedtest.count())
parsedtest.unpersist()
RMSE=math.sqrt(val_err)

print("Root Mean Squared Error Test= " + str(RMSE))

