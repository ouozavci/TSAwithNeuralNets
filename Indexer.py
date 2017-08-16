import re
import numpy as np

min_word_length = 3
max_word_length = 5
class Indexer:
    allIndex = {}
    negativeIndex = {}
    positiveIndex = {}

    index_term_capacity = 1000

    def __init__(self, negativeFileAddress, positiveFileAddress):
        self.nFile = open(negativeFileAddress, 'r', encoding="utf8")
        self.pFile = open(positiveFileAddress, 'r', encoding="utf8")

    def fm(self,word):
        if (len(word) < min_word_length):
            return None
        else:
            word = word[:max_word_length]
            word = word.lower()
            return word

    def createIndex(self):
        negativeStr = self.nFile.read()

        wordList = re.sub("[^\w]", " ", negativeStr).split()

        for word in wordList:
            word = self.fm(word)
            if word is not None:
                if word in self.negativeIndex:
                    self.negativeIndex[word] += 1
                else:
                    self.negativeIndex[word] = 1
                if word in self.allIndex:
                    self.allIndex[word] += 1
                else:
                    self.allIndex[word] = 1

        positiveStr = self.pFile.read()
        wordList = re.sub("[^\w]", " ", positiveStr).split()

        for word in wordList:
            word = self.fm(word)
            if word is not None:
                if word in self.positiveIndex:
                    self.positiveIndex[word] += 1
                else:
                    self.positiveIndex[word] = 1
                if word in self.allIndex:
                    self.allIndex[word] += 1
                else:
                    self.allIndex[word] = 1

        sorted_words = sorted(self.allIndex.items(),key=lambda x: x[1],reverse=True)

        final_index = []
        for key in sorted_words[:self.index_term_capacity]:
            final_index.append(key[0])


        final_index.sort()

        output = open("index.txt",'w+',encoding="utf8")
        #array = np.array([[key] for key in final_index])
        for i in range(len(final_index)):
            output.write(final_index[i])
            output.write("\n")


        #np.savetxt("index.txt",array)
        output.close()
print("Object created")

