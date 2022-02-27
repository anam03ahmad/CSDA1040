import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
message_list = pickle.load(open('.\message_list.pkl', 'rb'))
spamlabel_list = pickle.load(open('.\spamlabel_list.pkl', 'rb'))
msg_todisplay=pd.DataFrame(list(zip(message_list,spamlabel_list)),columns=['message','spam_label'])
#Count vectorizer for bag of words

def getBowTransformer(mgs):
  cv=CountVectorizer()
  bow_transformer=cv.fit(mgs)
  # Print total number of vocab words
  print(len(bow_transformer.vocabulary_))
  return bow_transformer
def getTFIDF(msg_bow):
  tfidf_transformer = TfidfTransformer().fit(msg_bow)
  messages_tfidf = tfidf_transformer.transform(msg_bow)
  print(messages_tfidf.shape)
  return messages_tfidf
st.title('Verifying SMS is spam or not')
message_selected=st.selectbox("Select a message:",msg_todisplay['message'])
get_msgindex = msg_todisplay.loc[(msg_todisplay['message']== "ok lar joking wif u oni")].index[0]
bow_transformer=getBowTransformer(message_selected)
bow = bow_transformer.transform(message_selected)
message_tfidf=getTFIDF(bow)
st.write(get_msgindex)