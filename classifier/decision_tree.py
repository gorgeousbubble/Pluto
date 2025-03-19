#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return(shannonEnt)

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

if __name__ == '__main__':
    myDataSet, myLabels = createDataSet()
    print("myDataSet:", myDataSet)
    print("myLabels:", myLabels)
    shannonEnt = calcShannonEnt(myDataSet)
    print("myDataSet Shannon Entropy:", shannonEnt)
    splitDataSet1 = splitDataSet(myDataSet, 0, 0)
    splitDataSet2 = splitDataSet(myDataSet, 1, 1)
    print("splitDataSet1:", splitDataSet1)
    print("splitDataSet2:", splitDataSet2)
    pass