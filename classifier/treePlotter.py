#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def createPlot():
    fig = matplotlib.pyplot.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = matplotlib.pyplot.subplot(111, frameon=False)
    plotNode('decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    matplotlib.pyplot.show()

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', ha='center', va='center', bbox=nodeType, arrowprops=arrow_args)

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]) == dict:
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0:'no', 1: {'flippers':{0: 'no', 1:'yes'}}}}, {'no surfacing':{0:'no', 1:{'flippers':{0: {'head':{0:'no', 1:'yes'}}, 1:'no'}}}}]
    return listOfTrees[i]

if __name__ == '__main__':
    print("Retrieve Tree 1:", retrieveTree(1))
    myTree = retrieveTree(0)
    print("Leafs number of myTree:", getNumLeafs(myTree))
    print("Depth of myTree:", getTreeDepth(myTree))
    createPlot()
    pass