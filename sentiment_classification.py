import pandas as pd
import os
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
'''
folder = 'aclImdb'
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()
for f in ('test', 'train'):
    for l in ('pos','neg'):
        path = os.path.join(folder,f,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
df.columns = ['review', 'sentiment']
df.to_csv('movie_data.csv', index=False, encoding='utf-8')
df.head()
'''
df = pd.DataFrame()
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)
x_train = df.loc[:19999, 'review'].values
y_train = df.loc[:19999, 'sentiment'].values
x_test = df.loc[30000:,'review'].values
y_test = df.loc[30000:, 'sentiment'].values

tokenizer_obj = Tokenizer()
total_reviews = x_test+x_train
tokenizer_obj.fit_on_texts(total_reviews)


#pad sequences
max_lenght = max([len(review.split())for review in total_reviews])

#define vocabulary size
vocab_size = len(tokenizer_obj.word_index)+1

x_train_tokens = tokenizer_obj.texts_to_sequences(x_train)
x_test_tokens = tokenizer_obj.texts_to_sequences(x_test)

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_lenght,padding='post')
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_lenght,padding='post')
print('--------------finish----------')



