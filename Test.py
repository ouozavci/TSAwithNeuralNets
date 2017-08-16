import Indexer as i
import Trainer as t

import numpy as np
import re

i = i.Indexer('negative.txt',"positive.txt")
i.createIndex()

t = t.Trainer(10)
t.train("negative.txt","positive.txt")

syn0 = np.loadtxt("syn0.txt")
testStr = "Arkadaşlar uyarayım;boşuna şu filmlere para vermeyin.Zaten filmde iş yok!yok ille de izlicem,sana ne kardeşim diyosanız;3-4 ay sabredin aha bi baktınız film televizyonda..."


wordsInTest = re.sub("[^\w]", " ", testStr).split()

innerIndex = {}
sample = {}

for word in wordsInTest:
    word = word[:4]
    word = word.lower()
    if (len(word) > 2):
        innerIndex[word] = 1

for word in allIndex:
    if word in innerIndex:
        sample[word] = 1
    else:
        sample[word] = 0

sampleMatrix = np.array([[sample[word] for word in sample]]).transpose()

l1 = nonlin(np.dot(sampleMatrix.T, syn0))


print("Output: ")
print(l1)