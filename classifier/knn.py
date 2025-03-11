#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import array

import numpy
import operator
import matplotlib
import matplotlib.pyplot as plt

from numpy.matlib import zeros


def genera_dating_test_data():
    numpy.random.seed(42)
    num_samples = 300
    age = numpy.random.randint(18, 46, num_samples)
    income = numpy.round(numpy.random.uniform(20, 150, num_samples), 1)
    interest_match = numpy.random.randint(50, 101, num_samples)
    labels = numpy.where(
        (interest_match > 70) & ((income > 80) | (age < 30)),
        1,
        0
    )
    noise_mask = numpy.random.rand(num_samples) < 0.1
    labels[noise_mask] = 1 - labels[noise_mask]
    data = numpy.column_stack((age, income, interest_match, labels))
    numpy.random.shuffle(data)
    header = "Age\tIncome\tInterestMatch\tLabel"
    numpy.savetxt(
        "datingTestData2.txt",
        data,
        delimiter="\t",
        fmt=["%d", "%.1f", "%d", "%d"],
        header=header,
        comments=""
    )


def create_data_set():
    group = numpy.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
    diffMat = numpy.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    soredDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlable = lables[soredDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    soredClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return soredClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    returnMat = numpy.zeros((numberOfLines, 3))
    classLabelsVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelsVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelsVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(dataSet.shape)
    m = dataSet.shape[0]
    normDataSet = dataSet - numpy.tile(minVals, (m, 1))
    normDataSet = normDataSet / numpy.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestData2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]: errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))

if __name__ == '__main__':
    datingDataMat, datingLabels = file2matrix("./datingTestData2.txt")
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], c=datingLabels)
    plt.show()
    pass