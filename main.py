import pandas as pd
import streamlit as st

from NILValueModel import predict_nil
from SchoolHometownGroups import get_schools, get_hometowns

menu = ["Home", "Data Exploration", "Prediction: NIL value"]
choice = st.sidebar.selectbox('Navigation', menu)
st.sidebar.markdown("Please use drop down to navigate to different pages")

if choice == 'Home':
    st.title('CSDA 1050 Group 2:')
    st.title('NIL Prediction Model')
    st.subheader(
        "This project will use a supervised machine learning approach to create a NIL(Name, Image and Likeness) valuation model to estimate:" +
        "either 1) the NIL value of a current or future NCAA hockey player" +
        "or 2) the NIL value based on different features or inputs")

    st.subheader("References:")
    st.subheader("Nilsson, J., Sibner, P. (2022). Elite Hockey Prospects. https://www.eliteprospects.com/")
    st.subheader(
        "NCAA Name, Image and Likeness Data – Athletes with Sponsorships and Endorsements, 2022. https://nilcollegeathletes.com/athletes")
    st.subheader("World Hockey Hub – World Rankings, 2022. https://worldhockeyhub.com/world-rankings/")
    st.subheader("NCAA – Men’s Ice Hockey Division 1, 2022. https://www.ncaa.com/stats/icehockey-men/d1")

    st.subheader("Fatima Anam Ahmad")
    st.subheader("Dnyanesh Bailoor")
    st.subheader("Nareshini Dookhy")
    st.subheader("Joey Santiago")
    st.subheader("Cheng Qian")



elif choice == 'Prediction: NIL value':

    st.title('NIL Value Prediction for Hockey Player')

    algorithm_selected = st.selectbox("Select an algorithm:",
                                      ['Extreme Gradient Boost Regressor Model', 'Random Forest Model'])

    # description
    st.write("The NIL value is predicted based on the following features:" +
             "  \n  \t  1. Weight in LBS" +
             "  \n  \t  2. Height in Inch" +
             "  \n  \t  3. Followers Instagram" +
             "  \n  \t  4. Followers Tiktok" +
             "  \n  \t  5. Followers Twitter" +
             "  \n  \t  6. School" +
             "  \n  \t  7. Hometown")

    # inputs
    weight_selected = st.number_input('Weight in KG:', min_value=30, max_value=500, value=50, step=1)
    height_selected = st.number_input('Height in Inch:', min_value=30, max_value=200, value=30, step=1)
    f_instagram_selected = st.number_input('Followers Instagram:', min_value=0, max_value=10000000, value=2000, step=1)
    f_tiktok_selected = st.number_input('Followers Tiktok:', min_value=0, max_value=10000000, value=0, step=1)
    f_twitter_selected = st.number_input('Followers Twitter:', min_value=0, max_value=10000000, value=0, step=1)
    school_selected = st.selectbox('School', get_schools())
    hometown_selected = st.selectbox('Hometown', get_hometowns())

    if st.button('Predict NIL value'):
        # create df with all inputs
        player_df = pd.DataFrame({
            'Height_in_Inches': [height_selected],
            'Weight_in_Lbs': [weight_selected],
            'Followers Instagram': [f_instagram_selected],
            'Followers Tiktok': [f_tiktok_selected],
            'Followers Twitter': [f_twitter_selected],
            'School': [school_selected],
            'Hometown': [hometown_selected]
        })

        prediction = predict_nil(player_df, hometown_selected, school_selected, algorithm_selected)

        st.write('The Predicted NIL Value: ', '{:,}'.format(prediction.flat[0]))

else:
    st.title("Data Exploration")
    menu_list = ['Height vs NIL', 'Weight vs NIL', 'Followers Instagram vs NIL',
                 'Followers Tiktok vs NIL', 'Followers Twitter vs NIL']
    menu = st.radio("Menu", menu_list)

    if menu == 'Height vs NIL':
        st.image('Visualizations/Height_vs_NIL.png')

    elif menu == 'Weight vs NIL':
        st.image('Visualizations/Weight_vs_NIL.png')

    elif menu == 'Followers Instagram vs NIL':
        st.image('Visualizations/InstagramF_vs_NIL.png')

    elif menu == 'Followers Tiktok vs NIL':
        st.image('Visualizations/TiktokF_vs_NIL.png')

    elif menu == 'Followers Twitter vs NIL':
        st.image('Visualizations/TwitterF_vs_NIL.png')
