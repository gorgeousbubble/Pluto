#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import array
import os

import numpy
import operator
import matplotlib
import matplotlib.pyplot as plt
from numpy.ma.core import choose

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

def generate_test_digits(output_dir="testDigits", samples_per_class=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    templates = {
        0: [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        1: [[0, 1, 0], [1, 1, 0], [0, 1, 0]],
        2: [[1, 1, 1], [0, 0, 1], [1, 1, 1]],
        3: [[1, 1, 1], [0, 1, 1], [1, 1, 1]],
        4: [[1, 0, 1], [1, 1, 1], [0, 0, 1]],
        5: [[1, 1, 1], [1, 0, 0], [1, 1, 1]],
        6: [[1, 1, 1], [1, 0, 0], [1, 1, 1]],
        7: [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
        8: [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        9: [[1, 1, 1], [1, 1, 1], [0, 0, 1]]
    }
    for digit in range(10):
        for i in range(samples_per_class):
            matrix = numpy.zeros((32, 32))
            start_row = numpy.random.randint(25)
            start_col = numpy.random.randint(25)
            for r in range(3):
                for c in range(3):
                    val = templates[digit][r][c]
                    if numpy.random.random() < 0.2:
                        val = 1 - val
                    matrix[start_row + r][start_col + c] = val
            filename = f"{digit}_{i}.txt"
            with open(os.path.join(output_dir, filename), 'w') as f:
                for row in matrix:
                    f.write(''.join(map(str, row.astype(int))) + '\n')

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

def classifyPerson():
    resultList = ["not at all", "in small doses", "in large doses"]
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent filter miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestData2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = numpy.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult-1])

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, i*32+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = numpy.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if classifierResult != testFileList[i]: errorCount += 1.0
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is: %f" % (errorCount / float(mTest)))

if __name__ == '__main__':
    testVector = img2vector('testDigits.txt')
    print(testVector)
    pass