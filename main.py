from SpamOrHamModel import st, predict_spam_ham, get_model_eval
import seaborn as sns
import matplotlib.pyplot as plt

menu = ["Home", "Data Exploration", "Prediction: Spam or Ham", "Model Evaluations"]
choice = st.sidebar.selectbox('Navigation', menu)
st.sidebar.markdown("Please use drop down to navigate to different pages")

if choice == 'Home':
    st.title('CSDA 1040 Group 2:')
    st.title('Spam Detection Model')
    st.header(
        "In this project, our model determines if an sms is spam or not. We are using Natual Language Processing and the Multinomial Naive Bayes algorithm." +
        "We used the ham and spam message data from")
    st.subheader("https://archive.ics.uci.edu/ml/datasets/sms+spam+collection")

    st.subheader("Lakshmi Sameera Vemula")
    st.subheader("Cheng Qian")
    st.subheader("Joey Santiago")
    st.subheader("Nareshini Dookhy")
    st.subheader("Fatima Anam Ahmad")



elif choice == 'Prediction: Spam or Ham':

    st.title('Verifying SMS is spam or not')

    message_selected = st.text_input("Please enter a message:")
    algorithm_selected=st.selectbox("Select an algorithm:", ['SVC', 'Naive Bayes', 'KNN', 'Decision Tree', 'Logistic Regression', 'Random Forest'])

    if st.button('Ham or Spam?'):
        st.write("Your message: ", message_selected)
        st.write("Your algorithm: ", algorithm_selected)
        prediction = predict_spam_ham([message_selected], algorithm_selected)

        col1, col2 = st.columns([8, 6])

        if (prediction.flat[0] == 'ham'):
            with col1:
                st.markdown('### Ham! Looks like the message is good!')

            with col2:
                st.image('.\safe.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB",
                         output_format="auto")
        else:

            with col1:
                st.markdown('### Spam! Looks like spam. Be careful!')

            with col2:
                st.image('.\warning.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB",
                         output_format="auto")

elif choice == 'Data Exploration':
    st.title("Data Exploration")
    menu_list = ['Ham / Spam Word Freq by Size', 'Proportion of Ham vs Spam', 'Message length by Ham vs Spam']
    menu = st.radio("Menu", menu_list)

    if menu == 'Ham / Spam Word Freq by Size':
        st.header('Ham Word Frequency')
        st.image('.\Visualizations\ham freq - size.png')
        st.header('Spam Word Frequency')
        st.image('.\Visualizations\spam freq - size.png')

    elif menu == 'Proportion of Ham vs Spam':
        st.header('Pie Chart: Ham vs. Spam')
        st.image('.\Visualizations\pie chart.png')
        st.header('Bar Chart: Ham vs Spam')
        st.image('.\Visualizations\histgram.png')

    else:
        st.header('Ham vs. Spam Message length')
        st.image('.\Visualizations\ham spam msg len.png')

else:
    st.title("Model Evaluations")
    menu_list = ['SVC', 'Naive Bayes', 'KNN', 'Decision Tree', 'Logistic Regression', 'Random Forest']
    algorithm = st.selectbox("Algorithms", menu_list)

    classification_report = get_model_eval(algorithm)
    st.write(classification_report)