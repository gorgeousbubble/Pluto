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

if __name__ == '__main__':
    createPlot()
    pass