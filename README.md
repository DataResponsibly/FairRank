Overview of the FairRank package
--------------------------------

FairRank is a package that quantifies bias in rankings produced by a
score-based ranker, and mitigates that bias using an optimization
procedure.

FairRank includes python scipts to:

(1) generate synthetic rankings while controlling the degree of bias;
(2) compute bias in a ranking according to different fairness
measures;
(3) generate a new ranking that is as close as possible to the
original ranking, yet has lower bias.

Core code modules
-----------------
- dataGenerator.py contains the core code of the ranking generation
algorithm
- measures.py contains the fairness and accuracy measures
- optimization.py implements the optimization process
- utility.py includes data transformation and ranking score generator
code

Usage of the core code modules
------------------------------
runOptimization.py, runRealDataExp.py and runSyntheticExp.py usage the
above core code; these scripts can be invoked on the command line

Test code 
----------
- testDataGenerator tests the usage of dataGenerator.py
- testMeasures tests the usage of measures.py
- testOptimization tests the usage of runOptimization.py
- testRealdataExp tests the usage of runRealDataExp.py
- testSyntheticExp tests the usage of runSyntheticExp.py
- testUtility tests the usage of utility.py

The above test scripts include the usage guideline of core code.

Datasets
--------
Some of real data sets we used are includes inside folder datasets.

Documentation
-------------
Related documents are included inside the docs folder.
To cite this work, use https://arxiv.org/abs/1610.08559

