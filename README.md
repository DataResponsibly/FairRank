Usage of FairRank

FairRank includes python scipts for compute different fariness meaures, ranking-generate algorithm, optimization of simple score based ranking. 

dataGenerator.py contains core code of ranking-generate algorithm.

measures.py contains core code of fairness meaures and accuracy measures of rankings.

optimization.py contains core code of optimization process.

utility.py includes data transformation and ranking score generator.

runOptimization.py, runRealDataExp.py and runSyntheticExp.py are the usage of above core code that can be called through command line.

Test code of above script:

testDataGenerator tests the usage of dataGenerator.py.

testMeasures tests the usage of measures.py.

testOptimization tests the usage of runOptimization.py.

testRealdataExp tests the usage of runRealDataExp.py.

testSyntheticExp tests the usage of runSyntheticExp.py.

testUtility tests the usage of utility.py.

Above test scripts include the usage guideline of core code. Can be fitted into new data by changing corresponding parameters.

Real data:

Some of real data sets we used are includes inside folder datasets.

Docs:

Related documents are included inside folder docs.

