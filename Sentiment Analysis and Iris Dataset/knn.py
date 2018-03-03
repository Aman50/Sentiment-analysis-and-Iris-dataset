""" kNN Algorithm implementation for classification of iris flowers
    Author: Aman50
"""

import csv
import random
import math
import operator

#Loading the dataset
#We will pass data to the algorithm in the format : float float float float string
def loadDataset(filename, split, trainset=[], testset=[]):
    with open(filename,"rb") as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for rows in range(0,len(dataset)-1):
            for y in range(0,4):
                dataset[rows][y]=float(dataset[rows][y])
        random.shuffle(dataset)
        trainset.extend(dataset[:split])
        testset.extend(dataset[split:])

#Function for finding the Euclidean-Distance of the given instance and available instance
def euclideanDistance(instance1, instance2 , length):
    distance=0
    for i in range(0,length-1):
        distance+=(float(instance1[i])-float(instance2[i]))**2
    distance=math.sqrt(distance)
    return distance

#Function for finding k nearest neighbours based on the Euclidean-Distance
def drawNeighbour(trainset, test ,k):
    distances=[]
    for i in range(0,len(trainset)):
        dist=euclideanDistance(trainset[i],test,len(test)-1)
        distances.append((trainset[i],dist))
    distances.sort(key=operator.itemgetter(1))
    kNeighbour=[]
    for j in range(0,k):
        kNeighbour.append(distances[j][0])
    return kNeighbour

#Function to Classify based on the Neighbours selected
def classify(neighbours):
    counts={}
    for i in range(0,len(neighbours)):
        name=neighbours[i][-1]
        if name in counts:
            counts[name]+=1
        else:
            counts[name]=1
    counts=sorted(counts.iteritems(), key=operator.itemgetter(1) , reverse=True)
    return counts[0][0]

#Function to check the accuracy of the implmentation
def findAccuracy(testset, predictions):
    correct=0.0
    for i in range(0,len(predictions)):
        if cmp(predictions[i],testset[i][-1])==0:
            correct+=1
    accuracy=correct/len(predictions)*100.0
    return accuracy

#Finally,the main function which uses all above functions and completes the algorithm
def main():
    trainset=[]
    testset=[]
    prediction=[]
    result=-1
    loadDataset("iris.data", 67, trainset, testset)
    for i in range(0,len(testset)):
        knn=drawNeighbour(trainset, testset[i], 4)
        prediction.append(classify(knn))
        print "Actual: "+testset[i][-1]+", Predicted: "+prediction[i]
    result=findAccuracy(testset,prediction)
    print "The accuracy is "+str(result) +"%"

main()
