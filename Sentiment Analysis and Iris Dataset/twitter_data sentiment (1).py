# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 11:51:03 2017

@author: Aman
"""
import nltk.classify.util
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from string import punctuation

def required_form(sentence):
    tokens=word_tokenize(sentence)
    required={}
    for i in tokens:
        i=i.lower()
        if i not in stopwords.words("english") and i not in list(punctuation) and i not in [";",")","(",":","O","p","P","o"]:
            required[i]="True"
    return required
sentences=twitter_samples.strings("positive_tweets.json")
positive=[]
count=0
for sentence in sentences:
    positive.append((required_form(sentence),"positive"))
    count+=1
    print count

sentences=twitter_samples.strings("negative_tweets.json")
negative=[]
for sentence in sentences:
    negative.append((required_form(sentence),"negative"))
    count+=1
    print count

train_set=positive[:4000]+negative[:4000]
test_set=positive[4000:]+negative[4000:]

classifier=NaiveBayesClassifier.train(train_set)

print nltk.classify.util.accuracy(classifier,test_set)*100

print classifier.classify(required_form("The movie is not too good.The characters made a not good mark and didn't boosted its ratings.Not a Decent watch.Not recommended!"))