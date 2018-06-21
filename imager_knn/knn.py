# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 19:35:40 2018

@author: Koudura Konan
"""

import numpy as np
import math

class Knn:
    def __init__(self):
        pass
    
    
    def train(self,trainingData, trainingLabel):
        self.trainingData = trainingData
        self.trainingLabel = trainingLabel
        
    
    def min(self, x,k):
        out = {}
        for i in range(k):
            mn = math.inf * -1
            index = -1
            for j in range(len(x)):
                if(mn > x[j]):
                    mn = x[j]
                    index = j
            key = self.trainingLabel[index]
            if(key in out):
                out[key] += math.pow(10,len(str(out[key]))) + k-1-i
            else:
                out[key] = k-1-i
                
        return out
    
    
    def test(self, testData,k=1):
        i = 0
        Y = np.zeros(len(testData))
        for data in testData:
            distances = np.sum(np.abs(self.trainingData - data),1)
            labels = self.min(distances,k)
            scores = list(labels.values())
            Y[i] = list(labels.keys())[scores.index(max(scores))]
            i += 1
        return Y
