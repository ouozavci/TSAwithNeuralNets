import Indexer as i
import Trainer as t
import Classifier as c
import numpy as np
import re

i = i.Indexer('negative.txt',"positive.txt")
i.createIndex()

t = t.Trainer(500)
t.train("negative.txt","positive.txt")


c = c.Classifier()

print("TESTING......")
comments = open("negative_test.txt",'r',encoding="utf8").readlines()

total_score = 0.0
pos_score = 0
neg_score = 0
total_comment_count = 0
total_comment_count += len(comments)
for i in range(len(comments)):
    score = c.classify(comments[i])
    if score<0.5:
        total_score+= 1
        neg_score+=1

print("Score(Negative): ",(neg_score/len(comments)))

comments = open("positive_test.txt",'r',encoding="utf8").readlines()
total_comment_count += len(comments)
for i in range(len(comments)):
    score=c.classify(comments[i])
    if score>0.5:
        total_score+= 1
        pos_score+=1

print("Score(Positive): ",(pos_score/len(comments)))

total_score = total_score/total_comment_count

print("Testing completed!")
print("Total Score: ",total_score)

while True:
    comment = input("Yorum: ");
    score = c.classify(comment)
    if score > 0.55:
        print("positive :: ",score)
    elif score < 0.45:
        print("negative :: ",score)
    else:
        print("neutral :: ",score)
