# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:05:30 2021

@author: Gage
"""

import pandas as pd
import numpy as np
import re, nltk, spacy, gensim, os, glob
from textblob import TextBlob
from varname import nameof
import matplotlib.pyplot as plt
### Functions ###


### Tokenizer
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


### Lemmatization function
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']): #'NOUN', 'ADJ', 'VERB', 'ADV'
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out


### Import all Tweet CSVs

os.chdir(r"C:\Users\Gage\Desktop\Academic\NLP Project\Tweets")

path = os.getcwd()

csv_files = glob.glob(os.path.join(path,"*.csv"))


df=pd.DataFrame()

for f in csv_files:
    df1 = pd.read_csv(f,encoding=("utf-8"))
    df = pd.concat([df,df1],axis=0)


master_df = pd.DataFrame(df, columns = ['user','text'])




### Cleaning step
#################

# Convert to list
data = df.text.tolist()
# Remove Emails
data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]
# Remove new line characters
data = [re.sub(r'\s+', ' ', sent) for sent in data]
# Remove distracting single quotes
data = [re.sub(r"\'", "", sent) for sent in data]

data_words = list(sent_to_words(data))


### Lemmatization step

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'VERB']) #select noun and verb
    
biden = master_df[master_df['text'].str.contains('Biden')]
biden['word'] = 'biden'

trump = master_df[master_df['text'].str.contains('Trump')]
trump['word'] = 'trump'

jan6 = master_df[master_df['text'].str.contains('January 6')]
jan6['word'] = 'jan6'

russia = master_df[master_df['text'].str.contains('Russia')]
russia['word'] = 'russia'

china = master_df[master_df['text'].str.contains('China')]
china['word'] = 'china'

words = [biden, trump, jan6, russia, china]

sentiment_dict = pd.DataFrame()

for r in words:
    
    sentiment_objects = [TextBlob(tweet) for tweet in r['text']]


    sentiment_objects[0].polarity, sentiment_objects[0]

    sentiment_values = pd.DataFrame([[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects])

    df = df.loc[~df.index.duplicated(keep='first')]

    r['sentiment'] = sentiment_values[0]


for word in words:
    sentiment_dict[str(word['word'].iloc[0])] = word.groupby(word['user']).mean()


for word in words:
    
    _ = plt.bar(sentiment_dict.index.values, sentiment_dict[str(word['word'].iloc[0])])
    _ = plt.title(str(word['word'].iloc[0]))
    _ = plt.xticks(rotation=90)
    _ = plt.axhline(y=0, color = 'r', linestyle = '-')
    plt.show()

