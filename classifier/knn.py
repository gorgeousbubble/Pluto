#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import array

import numpy
import operator


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

if __name__ == '__main__':
    group, labels = create_data_set()
    print("group:", group)
    print("labels:", labels)
    result = classify0([0, 0], group, labels, 3)
    print("result:", result)
    pass