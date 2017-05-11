import nltk
import sqlite3
import random
import pandas as pd
import numpy as np
from nltk.classify.scikitlearn import SklearnClassifier

conn = sqlite3.connect('yelpHotelData.db')
query = 'SELECT reviewContent, rating, usefulCount, coolCount, funnyCount FROM review WHERE flagged = "Y"'
fake = pd.read_sql(query, conn)
query = 'SELECT reviewContent, rating, usefulCount, coolCount, funnyCount FROM review WHERE flagged = "N"'
real = pd.read_sql(query, conn)
conn.close()

fake = fake.sample(750)
fake['tag'] = 'fake'
real = real.sample(750)
real['tag'] = 'real'
df = pd.concat([fake,real])
df = df.iloc[np.random.permutation(len(df))]

df['reviewContent'] = df.apply(lambda row: nltk.word_tokenize(row['reviewContent']), axis=1)
all_words = nltk.FreqDist(word.lower() for row in df['reviewContent'] for word in row)
word_features = list(all_words)[:2000]

del(all_words)
del(real)
del(fake)

def document_features(doc):
    document_words = set(doc['reviewContent'])
    features = {}
    for word in word_features:
        features['contains ' + word] = (word in document_words)
    features.update(
    {'rating': doc['rating'], 'useful': doc['usefulCount'], 'cool': doc['coolCount'], 'funny': doc['funnyCount'], 'length': len(doc['reviewContent'])})
    return [features,doc['tag']]

featuresets = df.apply(document_features, axis = 1)
train_set, test_set = featuresets[250:], featuresets[:250]
del(word_features)
del(featuresets)

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(7)

from sklearn.svm import SVC
svmClass = SklearnClassifier(SVC(C = .7)).train(train_set)
print("SVM Classifier:")
print(nltk.classify.accuracy(svmClass, test_set))

from sklearn.ensemble import AdaBoostClassifier
adaClass = SklearnClassifier(AdaBoostClassifier()).train(train_set)
print("Adaboost Classifier:")
print(nltk.classify.accuracy(adaClass, test_set))

from sklearn.neural_network import MLPClassifier
nnClass = SklearnClassifier(MLPClassifier()).train(train_set)
print("Neural Network Classifier:")
print(nltk.classify.accuracy(nnClass, test_set))
