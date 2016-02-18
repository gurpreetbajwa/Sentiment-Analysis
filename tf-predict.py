__author__ = 'Dell'
from pyspark import SparkConf, SparkContext,SQLContext, Row
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.linalg import Vectors
import sys, math

conf = SparkConf().setAppName('tf-idf')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

train = sys.argv[1]
test = sys.argv[2]

# Load and parse the data
def parsePoint(line):
    return LabeledPoint(float(line[1]), line[0])

#train data
train_data = sc.pickleFile(train)
parsedData = train_data.map(parsePoint)
#test data
test_data = sc.pickleFile(test)
parsedtestData = test_data.map(parsePoint)

# cross validation
num_iterations = 100
step_size=[0.1,10,20,300]
best_error=1000000
best_model=[0]
best_step=0
best_test_error=0
best_split=[]
best_RMSE=0

splits=[[1,2],[1,3]]

for x in step_size:
    for y in splits:
            (train_RDD,valid_RDD)=train_data.randomSplit(y,20L)
            parsed_input=train_RDD.map(parsePoint)
            parsed_valid=valid_RDD.map(parsePoint)
            model = LinearRegressionWithSGD.train(parsed_input,iterations=num_iterations,step=x)
            valuesAndPreds = parsed_valid.map(lambda p: (p.label, model.predict(p.features)))
            mse = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
            rmse=math.sqrt(mse)

            if rmse<best_error:
                best_step=x
                best_error=rmse

    # print split_array
    model = LinearRegressionWithSGD.train(parsedData,iterations=num_iterations,step=best_step)
    valuesAndPreds = parsedtestData.map(lambda p: (p.label, model.predict(p.features)))
    mse_test = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
    rmse_test=math.sqrt(mse_test)
    #
    print("Best Root Mean Squared Error in Cross-Val : " + str(best_error))
    print("Best Root Mean Squared Error on Test: " + str(rmse_test))
    print("Best Step size : " + str(best_step))

