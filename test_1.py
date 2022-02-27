from Assignment2_app import st, msg_todisplay, getBowTransformer, getTFIDF, predict_spam_ham

menu = ["Home", "Data Exploration", "Prediction: Spam or Ham"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home':
    st.sidebar.markdown("Home")
    st.title('CSDA 1040 Group 2:')
    st.title('Spam Detection Model')
    st. header("In this project, our model determines if an sms is spam or not. We are using Natual Language Processing and the Multinomial Naive Bayes algorithm."+
                  "We used the ham and spam message data from")
    st.subheader("https://archive.ics.uci.edu/ml/datasets/sms+spam+collection")

    st.subheader("Lakshmi Sameera Vemula")
    st.subheader("Cheng Qian")
    st.subheader("Joey Santiago")
    st.subheader("Nareshini Dookhy")
    st.subheader("Fatima Anam Ahmad")



elif choice == 'Prediction: Spam or Ham':
    st.sidebar.markdown("Model is trained on the Message Data")


    st.title('Verifying SMS is spam or not')

    message_selected = st.text_input("Please enter a message:")
    # message_selected=st.selectbox("Select a message:",msg_todisplay['message'])

    if st.button('Ham or Spam?'):
        st.write("Your message: ", message_selected)
        st.write(predict_spam_ham([message_selected]))

else:
    st.sidebar.markdown("Exploration")