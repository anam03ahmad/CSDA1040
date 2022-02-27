import streamlit as st
import pickle
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

message_list = pickle.load(open('.\message_list.pkl', 'rb'))
spamlabel_list = pickle.load(open('.\spamlabel_list.pkl', 'rb'))
msg_todisplay = pd.DataFrame(list(zip(message_list, spamlabel_list)), columns=['message', 'spam_label'])


# Count vectorizer for bag of words

# Bag of words transformer
def getBowTransformer(mgs):
    cv = CountVectorizer()
    bow_transformer = cv.fit(mgs)
    return bow_transformer


# vecortizer
def getTFIDF(msg_bow):
    tfidf_transformer = TfidfTransformer().fit(msg_bow)
    messages_tfidf = tfidf_transformer.transform(msg_bow)
    return messages_tfidf


# the training and test sets have to have the same shape
# the shape is determined when transforming
# append the new message to the end of messages
# transform data
# return the last row of transformed data as thatrepresents the test case
def transform_train_test(msg):
    ser_msg = msg_todisplay['message']

    ser_new_msg = pd.Series(msg)
    print("shape new msg: ", ser_new_msg.shape)

    ser_msg = ser_msg.append(ser_new_msg)
    print("Initial shape: ", ser_msg.shape)

    train_bow_transformer = getBowTransformer(ser_msg)
    train_messages_bow = train_bow_transformer.transform(ser_msg)
    train_messages_tfidf = getTFIDF(train_messages_bow)

    print("Resulting shape: ", train_messages_tfidf.shape)
    print(type(train_messages_tfidf))

    #last_row = len(ser_msg) - 1
    #test_row = train_messages_tfidf.getrow(last_row)
    test_row = train_messages_tfidf[-1:]

    train_messages_tfidf = train_messages_tfidf[0:-1]

    print('resulting test shape', test_row.shape)
    print('resutling train shape:', train_messages_tfidf.shape)

    return test_row, train_messages_tfidf


# returns new multinomial naive bayes model
# after training
def training(tfidf, label_results):
    mnb = MultinomialNB().fit(tfidf, label_results)
    return mnb


# get the model that uses the message data
def get_model(train_messages_tfidf):

    msg_train, msg_test, label_train, label_test = train_test_split(train_messages_tfidf, msg_todisplay['spam_label'],
                                                                    test_size=0.25)

    spam_detect_model = training(msg_train, label_train)

    return spam_detect_model


# predict if a message is ham or spam
def predict_spam_ham(msg):

    test_msg, train_data = transform_train_test(msg)

    model = get_model(train_data)

    prediction = model.predict(test_msg)

    result = 'Spam! Looks like spam. Be careful!'
    if (prediction.flat[0] == 'ham'):
      result = 'Ham! Looks like the message is good!'

    return result
