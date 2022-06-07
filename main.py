import pandas as pd
import streamlit as st

from GetData import get_random_comment

from flair.models import TextClassifier
from flair.data import Sentence

menu = ["Home", "Data Trends", "Comment Analysis Model"]
choice = st.sidebar.selectbox('Navigation', menu)
st.sidebar.markdown("Please use drop down to navigate to different pages")


def display_output(message_selected):
    sia = TextClassifier.load('en-sentiment')

    sentence = Sentence(message_selected)
    sia.predict(sentence)
    score = sentence.labels[0]

    if "POSITIVE" in str(score):
        img = 'Images/happy.png'
        msg = 'This comment is positive.'

    elif "NEGATIVE" in str(score):
        img = 'Images/angry.png'
        msg = 'This comment is negative.'

    else:
        img = 'Images/neutral.png'
        msg = 'This comment is neutral.'

    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        st.image(img)

    with col2:
        st.write(msg)


if choice == 'Home':
    st.title('Data Scientist (Research & Insights) – R-1003740')
    st.subheader('Anam Ahmad:')
    st.subheader('NLP Demo App')
    st.subheader(
        "This is a demo app designed to illustrate how machine learning can analyze sentiments of open-ended comments"
        + " using NLP and provide different visualizations of structured and comment data. I have developed this"
        + " with open source data. For the comments, I have taken data of various Youtube comments and focused on the"
        + " English ones. For the data visualizations, I have chosen to use the Youtube data for the word analysis, and"
        + " a 2013 US census data.")

    st.subheader("References:")
    st.subheader(
        "Github. Jolly, Mitchell. Trending-Youtube-Scraper (2018). https://github.com/mitchelljy/Trending-YouTube-Scraper")
    st.subheader(
        "American Community Survey (2022). https://www2.census.gov/programs-surveys/acs/data/pums/2013/1-Year/")

    st.subheader("Fatima Anam Ahmad")
    st.subheader("All rights reserved.")



elif choice == 'Comment Analysis Model':

    st.title('Sentiment Analysis for Comments')

    # description
    st.write("Try out with a comment! Enter your own comment or press 'random' to generate a random comment.")

    # inputs
    message_selected = st.text_input("Please enter a comment:")

    if st.button('Random Comment'):
        message_selected = get_random_comment()
        st.write(message_selected)

        display_output(message_selected)

    if st.button('Analyse Sentiment'):
        st.write('Comment:\n' + message_selected)
        display_output(message_selected)


else:
    st.title("Data Trends")
    menu_list = ['Word Analysis At a Glance', 'Avg Internet Access By State', 'FoodStamp Average By State',
                 'Income By State', 'Income By Age and Nativity']
    menu = st.radio("Menu", menu_list)

    if menu == 'Word Analysis At a Glance':
        st.image('Visualizations/wordcloud.png')
        st.caption('Key words from the Yt data are analyzed and the sized displayed according to their occurence. From a '
                'glance, we can see that "Love" is one of the most used words and there aren\'t any negative words '
                'that show up which implies that a large number of the comments are positive.')

    elif menu == 'FoodStamp Average By State':
        st.image('Visualizations/fs_avg.png')
        st.caption('Food Stamps by State. 1 refers to none of the populaion is on Food Stamps, and 2 refers to the '
                'entire population on Food Stamp. When a state is closer to 1, the less of the population is on Food '
                'Stamps. In contrast, the closer to 2, the more the more its population is on Food Stamps.')

    elif menu == 'Avg Internet Access By State':
        st.image('Visualizations/access_avg.png')
        st.caption('Internet Access by State. ')

    elif menu == 'Income By State':
        st.image('Visualizations/income_med.png')
        st.caption("The states are organized by their median family income from most to least. The income is divided by "
                "10000 as to standardize the values.")

    else:
        st.image('Visualizations/inc_age_nat.png')
        st.caption("This is tne family income of individuals by age for both Native born (blue) and Foreign born (orange).")

