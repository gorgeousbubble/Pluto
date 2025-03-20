#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

class Knn:
    def __init__(self, k=5, category="classifier"):
        self.y_train = None
        self.x_train = None
        self.k = k
        self.category = category

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        dist = torch.cdist(x_test, self.x_train, p=2)
        _, indices = torch.topk(dist, self.k, largest=False, dim=1)
        labels = self.y_train[indices]
        if self.category == "classifier":
            prediction = torch.mode(labels, dim=1).values
            return prediction
        elif self.category == "regressor":
            prediction = torch.mean(labels, dim=1)
            return prediction

if __name__ == '__main__':
    x_train = torch.randn(100, 5)
    y_train = torch.randint(0, 3, (100,))
    x_test = torch.randn(10, 5)
    knn = Knn(k=3, category="classifier")
    knn.fit(x_train, y_train)
    prediction = knn.predict(x_test)
    print("x_train:", x_train)
    print("y_train:", y_train)
    print("x_test:", x_test)
    print("prediction:", prediction)
    pass