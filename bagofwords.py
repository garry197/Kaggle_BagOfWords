# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:45:18 2018

@author: Garry
"""

import numpy as np
import pandas as pd

train=pd.read_csv('labeledTrainData.tsv',delimiter='\t')
test=pd.read_csv('testData.tsv',delimiter='\t')

final=[]
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from KaggleWord2VecUtility import KaggleWord2VecUtility

for i in range(0,len(train['review'])): 
  final.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], True)))



final_test=[]
for i in range(0,len(test['review'])):
  final_test.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], True)))





from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='word',stop_words=None,max_features=10000)
x=cv.fit_transform(final_test).toarray()



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='word',stop_words=None,max_features=10000)
train_x=cv.fit_transform(final).toarray()


y=train.drop('review',axis=1)
y.drop('id',axis=1,inplace=True)

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest.fit(train_x,train['sentiment'])

result = forest.predict(x)


submission=pd.DataFrame({'id':test['id'],'sentiment':result})

submission.to_csv('Submission.csv',index=False)
