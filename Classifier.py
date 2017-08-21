import re
import numpy as np

from Indexer import max_word_length
from Indexer import min_word_length

class Classifier:

    #def nonlin(self,x, deriv=False):
    #    if deriv == True:
    #        return x * (1 - x)
    #    return 1 / (1 + np.exp(-x))

    def nonlin(self,x, deriv=False):
        if deriv == True:
            return np.exp(-x)/((1+np.exp(-x))*(1+np.exp(-x)))
        return 1 / (1 + np.exp(-x))

    def classify(self,comment):

        syn0 = np.loadtxt("syn0.txt")

        wordsInComment = re.sub("[^\w]"," ",comment).split()

        innerIndex = {}
        sample = {}

        allIndex = open("index.txt",'r',encoding="utf8").readlines()
        for i in range(len(allIndex)):
            allIndex[i] = allIndex[i].replace("\r", "").replace("\n", "")

        for word in wordsInComment:
            word = word[:max_word_length]
            word = word.lower()
            if(len(word)>=min_word_length):
                innerIndex[word] = 1
        count = 0
        for word in allIndex:
            if word in innerIndex:
                sample[count] = 1
            else:
                sample[count] = 0
            count+=1

        sampleMatrix = np.array([sample[word] for word in sample]).transpose()

        l1 = self.nonlin(np.dot(sampleMatrix.T,syn0))

        return l1