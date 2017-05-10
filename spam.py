import nltk
import sqlite3
import random

conn = sqlite3.connect('yelpHotelData.db')
c = conn.cursor()

fake = []
real = []
for row in c.execute('SELECT reviewContent, rating, usefulCount, coolCount, funnyCount FROM review WHERE flagged = "Y" '):
    fake.append([nltk.word_tokenize(row[0]), row[1], row[2], row[3], row[4],'fake'])
random.shuffle(fake)
fake = fake[:500]

for row in c.execute('SELECT reviewContent, rating, usefulCount, coolCount, funnyCount FROM review WHERE flagged = "N" '):
    real.append([nltk.word_tokenize(row[0]), row[1], row[2], row[3], row[4],'real'])
random.shuffle(real)
real = real[:500]
documents = real + fake
random.shuffle(documents)

all_words = nltk.FreqDist(word.lower() for (doc,rt,use,cool,fun,tg) in documents for word in doc)
word_features = list(all_words)[:2000]

def document_features(doc,rt,use,cool,fun):
    document_words = set(doc)
    features = {}
    for word in word_features:
        features['contains ' + word] = (word in document_words)
    features.update({'rating': rt, 'useful': use, 'cool': cool, 'funny': fun, 'length': len(doc)})
    return features

featuresets = [(document_features(d,rt,use,cool,fun), c) for (d,rt,use,cool,fun,c) in documents]
train_set, test_set = featuresets[200:], featuresets[:200]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(7)
