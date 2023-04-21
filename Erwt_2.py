import numpy as np
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import keras

LIMIT=200

# Load data
df = pd.read_csv('spam_or_not_spam.csv')
df.drop(index=1466, axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

# Model of Neutral Network
model = keras.Sequential(name="Spam_mails")

model.add(keras.layers.LSTM(60, name='LSTM', input_shape=(100, LIMIT)))
model.add(keras.layers.Dense(32, name='Dense1', activation='relu'))
model.add(keras.layers.Dropout(0.3, name='Dropout1'))
model.add(keras.layers.Dense(16, name='Dense2', activation='relu'))
model.add(keras.layers.Dropout(0.1, name='Dropout2'))
model.add(keras.layers.Dense(1, name='Output', activation="sigmoid"))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=["binary_accuracy"])

# Make list of words
words =[]
for mail in df.email:
    for word1 in mail.lower().split(): #for word1 in mail.lower().split():
        if word1 not in words:
            words.append(word1)
# GloVe transform word to vector (word embeddings)
with open("glove.6B.200d.txt", 'r', encoding="utf-8") as f:
    embeddings_dict = {}
    for line1 in f:
        line1 = line1.split()
        word2 = line1[0]
        if word2 in words:
            vector = np.asarray(line1[1:], "float32")
            embeddings_dict[word2] = vector #=list(map(float, line1[1:]))

# limit length of embeddings_dict
for i in range(len(df)):
    mail2=[]
    for word3 in df.email[i].lower().split():
        if word3 in embeddings_dict.keys():
            mail2.append(embeddings_dict[word3])
    if (len(mail2) >= LIMIT):
        mail2=mail2[:LIMIT]
    else:
        mail2 += [list(np.zeros(200))] * (LIMIT - len(mail2))
    df.email[i]=mail2


#x=df.email.values
#y=df.label.values

x = np.asarray(list(df.email)).astype('float32')
y = np.array(df.label)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

model.summary()

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=60,batch_size=32,verbose=1)

y_pred=(model.predict(x_test) >0.5).astype('int32')

print('\tf1_score: {0:.2f}%,\tprecision: {1:.2f}%,\trecall: {2:.2f}%'.format(
        f1_score(y_test, y_pred, average='weighted') * 100,
        precision_score(y_test, y_pred, average='weighted') * 100,
        recall_score(y_test, y_pred, average='weighted') * 100))
