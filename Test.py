import Indexer as i
import Trainer as t
import Classifier as c

import numpy as np
import re

i = i.Indexer('negative.txt',"positive.txt")
i.createIndex()

t = t.Trainer(10)
t.train("negative.txt","positive.txt")


c = c.Classifier()
#c.classify("Alahımm filme bakk:D 7 yıl önce izlediğim o korkunç filmlerden biri fakat o kadar gülmüş ve dalga geçmiştik ki neredeyse sinemadan dışarı atılıyorduk Ebru ve Ben:) Bir laf var tam bizim durumu özetleyen .....z kaçınılmazsa tadnı çıkar:)) biz de öyle yaptık.")

comments = open("negative_test.txt",'r').readlines()

total_score = 0.0
total_comment_count = 0
total_comment_count += len(comments)
for i in range(len(comments)):
    score = c.classify(comments[i])
    if score<0.5:
        total_score+= 1

comments = open("positive_test.txt",'r').readlines()
total_comment_count += len(comments)
for i in range(len(comments)):
    score=c.classify(comments[i])
    if score>0.5: total_score+= 1

total_score = total_score/total_comment_count

print("Score: ",total_score)