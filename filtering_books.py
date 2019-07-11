# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 13:41:53 2019

@author: ishab
"""


import pandas as pd
import os

directory = "small/"
data = pd.DataFrame(columns=['asin','average_rating','book_id','description','ratings_count', 'title', 'title_without_series', 'series'])
for filename in os.listdir(directory):   
  data_book = pd.read_json("small/" + filename, lines=True)
  data_book = data_book[['asin','average_rating','book_id','description','ratings_count', 'title', 'title_without_series', 'series']]
#  data_book = data_book[data_book['description']!=''] #drops books with NA
#  data_book_drop = data_book[data_book['description']==''] 
  data = pd.concat([data,data_book], axis=0)#data that was dropped to map with the users.
  print(str(data_book.shape) + " " + str(data.shape))
#  with open('small/'+ filename) as f:
#    for line in f:
#      dataContent = json.loads(line)
data.to_csv("books.csv",index=False)
nlpdata = pd.read_csv('NLP_UI.csv')
bookids = nlpdata.book_id.unique()
data_filtered = data.loc[data['book_id'].isin(bookids)]
data_filtered1 = data_filtered[data_filtered['description']!=''] #drops books with NA
data_filtered_drop = data_filtered[data_filtered['description']==''] 
data_filtered2 = data_filtered1[data_filtered1.astype(str)['series'] == '[]'] #keeps the standalone books
data_filtered_series_drop = data_filtered1[data_filtered1.astype(str)['series'] != '[]']
books_we_need = data_filtered2.drop(columns=['series'])
books_we_need.to_csv("books_we_need.csv", index=False)

nlpdata1 = nlpdata[~nlpdata['book_id'].isin(data_filtered_drop.book_id.unique())]
nlpdata2 = nlpdata1[~nlpdata1['book_id'].isin(data_filtered_series_drop.book_id.unique())]
nlpdata2.to_csv("NLPDATA_DROP_SERIES_DES.csv", index=False)

