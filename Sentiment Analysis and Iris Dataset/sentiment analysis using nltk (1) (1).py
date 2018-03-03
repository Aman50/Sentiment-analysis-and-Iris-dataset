""" Sentiment Analysis using NLTK Library's built-in Naive Bayes Classifier
    Author: Aman50
"""


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

"""Nltk library only accepts
data in format "word":True
So we will tokenize each
word and remove stopwords
and convert it into dict.
of given format to avoid
duplication.The below function
does this"""
def suitable_words(words):
    #words=word_tokenize(sentence)
    processed_text={}
    for word in words:
        if word.lower() not in stopwords.words("english"):
            processed_text[word.lower()]="True"
    return processed_text

"""Now we create both positive
and negative lists which contain
respective reviews.This is to
train the naive baye's classifier
(Class 12th)"""

count=0
negative_reviews=[]
for fileid in movie_reviews.fileids("neg"):
    words=movie_reviews.words(fileid)
    negative_reviews.append((suitable_words(words),"negative"))
    count+=1
    print count
    if count==400:
        break
count=0
positive_reviews=[]
for fileid in movie_reviews.fileids("pos"):
    words=movie_reviews.words(fileid)
    positive_reviews.append((suitable_words(words),"positive"))
    count+=1
    print count
    if count==400:
        break
"""Now creating Test and train samples"""
train_set=negative_reviews[:200]+positive_reviews[:200]
test_set=negative_reviews[200:]+positive_reviews[200:]

#Creating Naive Bayes Classifier
classifier=NaiveBayesClassifier.train(train_set)

accuracy=nltk.classify.util.accuracy(classifier,test_set)
print(accuracy*100)

#sentence="The movie is not too good.The characters made a not good mark and didn't boosted its ratings.Not a Decent watch.Not recommended!"
#wordz=word_tokenize(sentence)
#print classifier.classify(suitable_words(wordz))
