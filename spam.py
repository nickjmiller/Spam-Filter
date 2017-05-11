import nltk
import sqlite3
import random
import pandas as pd
import numpy as np

conn = sqlite3.connect('yelpHotelData.db')
query = 'SELECT reviewContent, rating, usefulCount, coolCount, funnyCount FROM review WHERE flagged = "Y" OR flagged = "YR"'
fake = pd.read_sql(query, conn)
query = 'SELECT reviewContent, rating, usefulCount, coolCount, funnyCount FROM review WHERE flagged = "N" OR flagged = "NR"'
real = pd.read_sql(query, conn)
conn.close()


fake = fake.iloc[np.random.permutation(5000)]
fake['tag'] = 'fake'
real = real.iloc[np.random.permutation(5000)]
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
train_set, test_set = featuresets[1000:], featuresets[:1000]
del(word_features)
del(featuresets)

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(7)
