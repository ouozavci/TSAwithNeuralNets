import re
import numpy as np
from string import digits

from Indexer import max_word_length
from Indexer import min_word_length

class Classifier:

    #def nonlin(self,x, deriv=False):
    #    if deriv == True:
    #        return x * (1 - x)
    #    return 1 / (1 + np.exp(-x))

    def fm(self, word):

        remove_digits = str.maketrans('', '', digits)
        word = word.translate(remove_digits)

        if (len(word) < min_word_length):
            return None
        else:
            word = word[:max_word_length]
            word = word.lower()


            word = word.replace("ü", "u")
            word = word.replace("ğ", "g")
            word = word.replace("ş", "s")
            word = word.replace("ç", "c")
            word = word.replace("ı", "i")
            word = word.replace("ö", "o")

            return word

    def nonlin(self,x, deriv=False):
        if deriv == True:
            return np.exp(-x)/((1+np.exp(-x))*(1+np.exp(-x)))
        return 1 / (1 + np.exp(-x))

    def classify(self,comment):

        syn0 = np.loadtxt("syn0.txt")

        wordsInComment = re.sub('[^A-Za-z0-9ğüşçıöĞÜİŞÇÖ]+', ' ', re.sub("[^\w]", " ", comment)).split()

        innerIndex = {}
        sample = {}

        allIndex = open("index.txt",'r',encoding="utf8").readlines()
        for i in range(len(allIndex)):
            allIndex[i] = allIndex[i].replace("\r", "").replace("\n", "")

        for word in wordsInComment:
            word = self.fm(word)
            if word is not None:
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