# sentiment-analysis
extra credit for NLP

requirements:
  python
  spacy
  spacy english data
  numpy
  pandas
  sklearn
  
usage:

  there are three blocks of code at the bottom, 2 of them are commented. They each run the code in different ways, but the       third is most useful for grading. I've kept the other two to show the progression of the code and how I used it.

  if only block 1 is left uncommented: 

    python3 classifier.py <trainfile>
  
    This runs a 80/20 split on the data in trainfile and then reports the results
    
  if only block 1 is left uncommented: 

    python3 classifier.py <trainfile> <testfile>
  
    This trains on trainfile and tests on testfile, printing the results
    
  if only block 1 is left uncommented: 

    python3 classifier.py <trainfile> <testfile>
  
    This uses trainfile to build the index vectors and then tests on testfile, but loads the models instead of training them.     Use this only if you have the two model files bayes.joblib and neural.joblib in the same directory.

features used:

  1) word occurrence count, excluding words that occurred 0 or 1 times in the test set
  2) bigram occurrence count, excluding bigrams that occurred 0 or 1 times in the test set
  3) subreddit

classifiers used:

  I used a neural model as well as a Bernouli Naive Bayes model, as one may outperform the other due to the test data size. 

Results:

  Some emotions were better with Naive Bayes, others better with the neural model. All statistics for individual emotions as well as overall are included below. These can also be reproduced by 

Bernouli Naive Bayes:

  anger accuracy: 72.45989304812834
  anger precision: 0.5803571428571429
  anger recall: 0.5371900826446281
  anger f1: 0.5579399141630901
  anticipation accuracy: 59.893048128342244
  anticipation precision: 0.6847826086956522
  anticipation recall: 0.5779816513761468
  anticipation f1: 0.6268656716417911
  disgust accuracy: 70.32085561497327
  disgust precision: 0.6153846153846154
  disgust recall: 0.7692307692307693
  disgust f1: 0.6837606837606838
  fear accuracy: 65.50802139037434
  fear precision: 0.7357142857142858
  fear recall: 0.5282051282051282
  fear f1: 0.6149253731343284
  joy accuracy: 80.21390374331551
  joy precision: 0.4090909090909091
  joy recall: 0.12857142857142856
  joy f1: 0.19565217391304343
  love accuracy: 91.17647058823529
  love precision: 0.42105263157894735
  love recall: 0.26666666666666666
  love f1: 0.326530612244898
  optimism accuracy: 56.149732620320854
  optimism precision: 0.33497536945812806
  optimism recall: 0.7010309278350515
  optimism f1: 0.4533333333333333
  pessimism accuracy: 64.1711229946524
  pessimism precision: 0.45394736842105265
  pessimism recall: 0.575
  pessimism f1: 0.5073529411764707
  sadness accuracy: 66.57754010695187
  sadness precision: 0.5679012345679012
  sadness recall: 0.3382352941176471
  sadness f1: 0.423963133640553
  surprise accuracy: 85.29411764705883
  surprise precision: 0.3125
  surprise recall: 0.10204081632653061
  surprise f1: 0.15384615384615385
  trust accuracy: 83.9572192513369
  trust precision: 0.5
  trust recall: 0.18333333333333332
  trust f1: 0.26829268292682923
  neutral accuracy: 87.43315508021391
  neutral precision: 0.2
  neutral recall: 0.07894736842105263
  neutral f1: 0.11320754716981134

  overall accuracy: 73.59625668449198
  overall precision: 0.5452196382428941
  overall recall: 0.4906976744186046
  overall f1: 0.5165238678090576

Multi-Layer Perceptron model:

  anger accuracy: 76.20320855614973
  anger precision: 0.6951219512195121
  anger recall: 0.47107438016528924
  anger f1: 0.5615763546798029
  anticipation accuracy: 58.02139037433155
  anticipation precision: 0.7479674796747967
  anticipation recall: 0.42201834862385323
  anticipation f1: 0.5395894428152493
  disgust accuracy: 73.79679144385027
  disgust precision: 0.7230769230769231
  disgust recall: 0.6025641025641025
  disgust f1: 0.6573426573426574
  fear accuracy: 63.36898395721925
  fear precision: 0.7636363636363637
  fear recall: 0.4307692307692308
  fear f1: 0.5508196721311476
  joy accuracy: 82.0855614973262
  joy precision: 0.7142857142857143
  joy recall: 0.07142857142857142
  joy f1: 0.12987012987012989
  love accuracy: 91.44385026737967
  love precision: 0.25
  love recall: 0.03333333333333333
  love f1: 0.058823529411764705
  optimism accuracy: 72.45989304812834
  optimism precision: 0.45588235294117646
  optimism recall: 0.31958762886597936
  optimism f1: 0.37575757575757573
  pessimism accuracy: 71.92513368983957
  pessimism precision: 0.5789473684210527
  pessimism recall: 0.4583333333333333
  pessimism f1: 0.5116279069767442
  sadness accuracy: 66.57754010695187
  sadness precision: 0.6037735849056604
  sadness recall: 0.23529411764705882
  sadness f1: 0.33862433862433866
  surprise accuracy: 86.09625668449198
  surprise precision: 0.38461538461538464
  surprise recall: 0.10204081632653061
  surprise f1: 0.16129032258064516
  trust accuracy: 83.9572192513369
  trust precision: 0.5
  trust recall: 0.23333333333333334
  trust f1: 0.3181818181818182
  neutral accuracy: 89.03743315508021
  neutral precision: 0.2857142857142857
  neutral recall: 0.05263157894736842
  neutral f1: 0.08888888888888889

  overall accuracy: 76.24777183600713
  overall precision: 0.6555555555555556
  overall recall: 0.36589147286821705
  overall f1: 0.4696517412935323
  
