import numpy as np
import re
from Indexer import min_word_length
from Indexer import max_word_length
class Trainer:

    def __init__(self,iteration):
        self.iteration = iteration

    def train(self,negativeFileAddress,positiveFileAddress):

        # reading index file and creating a list of terms
        terms = open("index.txt",'r',encoding="utf8").readlines()
        for i in range(len(terms)):
            terms[i] = terms[i].replace("\r","").replace("\n","")

        #syn0 = 2 * np.random.random((len(terms), 1)) - 1
        syn0 = np.zeros((len(terms),1)) + 0.5
        #train for negative
        with open(negativeFileAddress, encoding="utf8") as f:
            lines = f.readlines()
        for iter in range(self.iteration):
            print("Training for negative                iteration:: ",iter)
            for line in lines:
                wordsInLine = re.sub("[^\w]", " ", line).split()
                sample = {}
                innerIndex = {}
                for word in wordsInLine:
                    word = word[: max_word_length]
                    word = word.lower()
                    if (len(word) >= min_word_length):
                        innerIndex[word] = 1
                for word in terms:
                    if word in innerIndex:
                        sample[word] = 1;
                    else:
                        sample[word] = 0;

                sampleMatrix = np.array([[sample[word] for word in sample]]).transpose()
                l1 = self.nonlin(np.dot(sampleMatrix.T, syn0))
                y = np.array([0])
                l1_error = y - l1
                # multiply how much we missed by slope of the sigmoid (nonlin function)
                l1_delta = l1_error * self.nonlin(l1, True)

                # update weights
                syn0 += np.dot(sampleMatrix, l1_delta)
##########################################################################################
        #train for positive
        with open(positiveFileAddress, encoding="utf8") as f:
            lines = f.readlines()
        for iter in range(self.iteration):
            print("Training for positive                iteration:: ",iter)
            for line in lines:
                wordsInLine = re.sub("[^\w]", " ", line).split()
                sample = {}
                innerIndex = {}
                for word in wordsInLine:
                    word = word[:max_word_length]
                    word = word.lower()
                    if (len(word) >= min_word_length):
                        innerIndex[word] = 1
                for word in terms:
                    if word in innerIndex:
                        sample[word] = 1;
                    else:
                        sample[word] = 0;

                sampleMatrix = np.array([[sample[word] for word in sample]]).transpose()
                l1 = self.nonlin(np.dot(sampleMatrix.T, syn0))
                y = np.array([1])
                l1_error = y - l1
                # multiply how much we missed by slope of the sigmoid (nonlin function)
                l1_delta = l1_error * self.nonlin(l1, True)

                # update weights
                syn0 += np.dot(sampleMatrix, l1_delta)

        print("TRAINING COMPLETED.")
        np.savetxt("syn0.txt",syn0)
        print("syn0 file created successfully")

    def nonlin(self,x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))