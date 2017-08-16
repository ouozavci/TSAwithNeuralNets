import Indexer as i
import Trainer as t
import Classifier as c

import numpy as np
import re

#i = i.Indexer('negative.txt',"positive.txt")
#i.createIndex()

#t = t.Trainer(10)
#t.train("negative.txt","positive.txt")


c = c.Classifier()
c.classify("başarılı")