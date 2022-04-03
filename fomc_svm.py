!pip install tweet-preprocessor
import sklearn
import numpy as np
import pandas as pd
from numpy.random import RandomState

df = pd.read_csv('fomcRaw.csv', skiprows =[i for i in range(1,55)], index_col=0)

dfSort = df.head(61)
dfLabel = pd.read_csv('labelled.csv', index_col=0)

df = pd.concat([dfSort, dfLabel], axis=1)

train = df.sample(frac=0.5, random_state=rng)
test = df.loc[~df.index.isin(train.index)]

train.to_csv ('train.csv')
test.to_csv ('test.csv')

# training data
train = pd.read_csv("train.csv")

# test data
test = pd.read_csv("test.csv")

# remove special characters using the regular expression library
import re

#set up punctuations we want to be replaced
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

import preprocessor as p

# custum function to clean the dataset (combining tweet_preprocessor and reguar expression)
def clean_content(df):
  tempArr = []
  for line in df:
    # send to tweet_processor
    tmpL = p.clean(line)
    # remove puctuation
    tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases
    tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
    tempArr.append(tmpL)
  return tempArr

# clean training data
train_content = clean_content(train["Content"])
train_content = pd.DataFrame(train_content)

# append cleaned tweets to the training data
train["clean_content"] = train_content

# compare the cleaned and uncleaned tweets
train.head()

# clean the test data and append the cleaned tweets to the test data
test_content = clean_content(test["Content"])
test_content = pd.DataFrame(test_content)
# append cleaned tweets to the training data
test["clean_content"] = test_content

# compare the cleaned and uncleaned tweets
test.head()

from sklearn.model_selection import train_test_split

# extract the labels from the train data
y = train["MP Tightening"].values

# use 70% for the training and 30% for the test
x_train, x_test, y_train, y_test = train_test_split(train.clean_content.values, y, 
                                                    stratify=y, 
                                                    random_state=1, 
                                                    test_size=0.5, shuffle=True)
from sklearn.feature_extraction.text import CountVectorizer

# vectorize tweets for model building
vectorizer = CountVectorizer(binary=True, stop_words='english')

# learn a vocabulary dictionary of all tokens in the raw documents
vectorizer.fit(list(x_train) + list(x_test))

# transform documents to document-term matrix
x_train_vec = vectorizer.transform(x_train)
x_test_vec = vectorizer.transform(x_test)

from sklearn import svm
# classify using support vector classifier
svm = svm.SVC(kernel = 'linear', probability=True)

# fit the SVC model based on the given training data
prob = svm.fit(x_train_vec, y_train).predict_proba(x_test_vec)

# perform classification and prediction on samples in x_test
y_pred_svm = svm.predict(x_test_vec)

from sklearn.metrics import accuracy_score
print("Accuracy score for SVC is: ", accuracy_score(y_test, y_pred_svm) * 100, '%')
