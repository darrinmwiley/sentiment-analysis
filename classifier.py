#pip3 install spacy
#pip3 install numpy
#pip3 install pandas
#python3 -m spacy download en

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
import sys


tok = spacy.load('en')
def tokenize (text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

def get_data(filename):
    reviews = pd.read_json(filename).T

    #count number of occurences of each word
    counts = Counter()
    subreddits = Counter()
    bigrams = Counter()
    for index, row in reviews.iterrows():
        counts.update(tokenize(row['body']))
        subreddits.update(tokenize(row['subreddit']))
        tokens = tokenize(row['body'])
        for i in range(len(tokens)-1):
            bigrams.update([tokens[i]+"-"+tokens[i+1]])

    #deleting infrequent words
    print("num_words before:",len(counts.keys()))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    print("num_words after:",len(counts.keys()))

    print("subreddits before:",len(subreddits.keys()))
    for word in list(subreddits):
        if subreddits[word] < 2:
            del subreddits[word]
    print("subreddits after:",len(subreddits.keys()))

    print("bigrams before:",len(bigrams.keys()))
    for word in list(bigrams):
        if bigrams[word] < 2:
            del bigrams[word]
    print("bigrams after:",len(bigrams.keys()))

    #creating vocabulary
    vocab2index = {}
    indices = []
    for word in counts:
        vocab2index[word] = len(indices)
        indices.append(word)

    #creating vocabulary
    sub2index = {}
    for word in subreddits:
        sub2index[word] = len(indices)
        indices.append(word)

    #creating bigram vocabulary
    bi2index = {}
    for word in bigrams:
        bi2index[word] = len(indices)
        indices.append(word)

    Z = [[],[],[],[],[],[],[],[],[],[],[],[]]
    #              74       69             72        71     88       89      71          65            73         87        81        81
    emotions = ['anger','anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust', 'neutral']
    for index, row in reviews.iterrows():
        arr = [0]*len(indices)
        yarr = []
        tokens = tokenize(row['body'])
        for t in tokens:
            if t in vocab2index:
                arr[vocab2index[t]] = arr[vocab2index[t]]+1
        for i in range(len(tokens)-1):
            big = tokens[i]+"-"+tokens[i+1]
            if big in bi2index:
                arr[bi2index[big]] = arr[bi2index[big]]+1
        if row['subreddit'] in sub2index:
            arr[sub2index[row['subreddit']]] = 1
        for emo in emotions:
            yarr.append(1 if row['emotion'][emo] else 0)
        for i in range(len(emotions)):
            Z[i].append([np.array(arr), yarr[i]])
    return Z

def split_data(data):
    total = len(data[0])
    training_ratio = 0.8
    training_data = [[],[],[],[],[],[],[],[],[],[],[],[]]
    evaluation_data = [[],[],[],[],[],[],[],[],[],[],[],[]]

    for i in range(len(data)):
        for indice in range(0, total):
            if indice < total * training_ratio:
                training_data[i].append(data[i][indice])
            else:
                evaluation_data[i].append(data[i][indice])

    return training_data, evaluation_data

def preprocessing_step():
    data = get_data("train.txt")

    return split_data(data)

def train_test():
    print("beginning train/test")
    train = get_data(sys.argv[1])
    print("test data formatted. formatting training data.")
    test = get_data(sys.argv[2])
    print("data formatted, training classifiers")
    classifiers = training_step(train)
    print("classifiers trained, beginning evaluation")
    simple_evaluation(classifiers, test)

def training_step(data):
    ret = []
    for i in range(len(data)):
        training_text = [dat[0] for dat in data[i]]
        training_result = [dat[1] for dat in data[i]]
        ret.append(BernoulliNB().fit(training_text, training_result))
    return ret


def analyse_text(classifier, vector):
    return classifier.predict([vector])

def print_result(result):
    text, analysis_result = result
    print_text = "Positive" if analysis_result[0] == '1' else "Negative"
    print(text, ":", print_text)

def simple_evaluation(classifiers,evaluation_data):
    emotions = ['anger','anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust', 'neutral']
    total = len(evaluation_data[0])*12
    corrects = 0
    #actual,classified
    confusion = [[0.0,0.0],[0.0,0.0]]
    for i in range(len(classifiers)):
        local_confusion = [[0.0,0.0],[0.0,0.0]]
        evaluation_vector     = [ev[0] for ev in evaluation_data[i]]
        evaluation_result   = [ev[1] for ev in evaluation_data[i]]
        tot = len(evaluation_data[0])
        crr = 0
        for index in range(0, tot):
            analysis_result = analyse_text(classifiers[i], evaluation_vector[index])
            result = analysis_result
            #print(str(result[0])+", "+str(evaluation_result[index]))
            local_confusion[evaluation_result[index]][result[0]] = local_confusion[evaluation_result[index]][result[0]] + 1
            confusion[evaluation_result[index]][result[0]] = confusion[evaluation_result[index]][result[0]] + 1
            corrects += 1 if result[0] == evaluation_result[index] else 0
            crr += 1 if result[0] == evaluation_result[index] else 0
        print(emotions[i]+" accuracy: "+str(crr*100.0/tot))
        local_precision = local_confusion[1][1]/(local_confusion[1][1] + local_confusion[0][1])
        local_recall = local_confusion[1][1]/(local_confusion[1][1] + local_confusion[1][0])
        local_f1 = 2*(local_precision*local_recall)/(local_precision + local_recall)
        print(emotions[i]+" precision: "+str(local_precision))
        print(emotions[i]+" recall: "+str(local_recall))
        print(emotions[i]+" f1: "+str(local_f1))
    print("\n")
    precision = confusion[1][1]/(confusion[1][1] + confusion[0][1])
    recall = confusion[1][1]/(confusion[1][1] + confusion[1][0])
    f1 = 2*(precision*recall)/(precision + recall)
    print("overall accuracy: "+str(corrects * 100.0 / total))
    print("overall precision: "+str(precision))
    print("overall recall: "+str(recall))
    print("overall f1: "+str(f1))
    print()
    
print("beginning preprocessing")
training_data, evaluation_data = preprocessing_step()
print("fitting classifiers")
classifiers = training_step(training_data)
print("evaluating")
simple_evaluation(classifiers,evaluation_data)

#train_test()

