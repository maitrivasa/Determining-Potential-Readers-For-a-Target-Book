# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:40:38 2019

@author: ishab
"""

import re
import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy import spatial
import numpy as np


def preprocess(text):
  TAG_RE = re.compile(r'<[^>]+>')
  stop_words = set(stopwords.words('english')) 
  newText = TAG_RE.sub('', text)
  newText = re.sub("[^a-zA-Z ]+","", newText)
  word_tokens = word_tokenize(newText) 
  newerText = []
  for w in word_tokens: 
    if w not in stop_words: 
        newerText.append(w)
  lemmatizer = WordNetLemmatizer()
  lemmat=[]
  for i in newerText:
    lemmat.append(lemmatizer.lemmatize(i))
  
  return lemmat


def normalize(word_vec):
    norm=np.linalg.norm(word_vec)
    if norm == 0: 
       return word_vec
    return word_vec/norm
  
def train_model(df):
  cleanDesc = df['description'].apply(preprocess)
  cleanDesc = cleanDesc.tolist()
  dataForDoc2Vec = [TaggedDocument(words = doc, tags = str(i)) for i, doc in enumerate(cleanDesc)] 
  
  #TO DO FIGURE CORRECT VALUES
  max_epochs = 100
  vec_size = 100
  alpha = 0.025
  
  model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
  model.build_vocab(dataForDoc2Vec)
  #DO WE NEED EPOCHS??
  for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(dataForDoc2Vec,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
  model.save("d2v.model")
  print("Model Saved")

def find_similarity(book_description):
  global model
  target = normalize(model.infer_vector(target_book))
  past_book = normalize(model.infer_vector(book_description))
  similarity = spatial.distance.cosine(target, past_book)


def driver():
  df = pd.read_csv("comicBooks.csv")
  global target 
  target = df[df['book_id']== 472331]
  target = target.iloc[0,3]
  target = preprocess(str(target))
  df = df[df['book_id']!= 472331]
  train_model(df)
  global model
  model = Doc2Vec.load("d2v.model")
  print("Model Loaded")
  dfUI = pd.read_csv("NLP_Comics.csv")
  dfUI = dfUI[['user_id', 'book_id', 'rating', 'description']]
  dfUIOnlyDes = dfUI['description'].apply(preprocess)
  dfSim = dfUI['description'].apply(find_similarity)
  
driver()
  
  