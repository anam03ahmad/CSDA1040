import pandas as pd

from SurvivalModel import st, predict_suvival, get_model_eval
import seaborn as sns
import matplotlib.pyplot as plt

menu = ["Home", "Data Exploration", "Prediction: Survival Days", "Model Evaluations"]
choice = st.sidebar.selectbox('Navigation', menu)
st.sidebar.markdown("Please use drop down to navigate to different pages")


def fill_celltype_matrix(df, cell_selected):
    if cell_selected == 'Large':
        df.at[0, 'celltype_large'] = 1
        df.at[0, 'celltype_smallcell'] = 0
        df.at[0, 'celltype_squamous'] = 0
    elif cell_selected == 'Smallcell':
        df.at[0, 'celltype_large'] = 0
        df.at[0, 'celltype_smallcell'] = 1
        df.at[0, 'celltype_squamous'] = 0
    elif cell_selected == 'Squamous':
        df.at[0, 'celltype_large'] = 0
        df.at[0, 'celltype_smallcell'] = 0
        df.at[0, 'celltype_squamous'] = 1
    else:
        df.at[0, 'celltype_large'] = 0
        df.at[0, 'celltype_smallcell'] = 0
        df.at[0, 'celltype_squamous'] = 0
    return df


if choice == 'Home':
    st.title('CSDA 1040 Group 2:')
    st.title('Survival Prediction Model')
    st.header(
        "In this project, our model predicts how many days a given veteran lung cancer patient is to survive." +
        "Our data is built on a study with veterans with lung cancer and their survival." +
        "We are using the veteran lung data from")
    st.subheader("https://archive.ics.uci.edu/ml/datasets/veteran")

    st.subheader("Lakshmi Sameera Vemula")
    st.subheader("Cheng Qian")
    st.subheader("Joey Santiago")
    st.subheader("Nareshini Dookhy")
    st.subheader("Fatima Anam Ahmad")



elif choice == 'Prediction: Survival Days':

    st.title('Survival Days Prediction for Veteran')

    algorithm_selected=st.selectbox("Select an algorithm:", ['Cox Proportional-Hazards Model', 'Weibull Accelerated Failure Time Model'])

    #description
    st.write("The survival days is predicted based on the following attributes of a patient:" +
             "\n\t1. Treatment - Standard or Test" +
             "\n\t2. Cancer Cell Type - Squamous, Smallcell, Adeno, Large" +
             "\n\t3. Karnofsky Performance Score (100=good)" +
             "\n\t4. Diagnosis Time - Months from Diagnosis to Randomisation" +
             "\n\t5. Age" +
             "\n\t6. If they've had Prior Therapy")

    #inputs
    trt_selected = st.selectbox( 'Treatment Type', ('Standard', 'Test'))
    cell_selected = st.selectbox('Cancer Cell Type', ('Squamous', 'Smallcell', 'Adeno', 'Large'))
    karno_selected = st.number_input('Karnofsky Score', min_value=1, max_value=100, value=50, step=1)
    diag_selected = st.number_input('Diagnosis Time', min_value=1, max_value=100, value=50, step=1)
    age_selected = st.number_input('Age', min_value=30, max_value=100, value=65, step=1)
    prior_selected = st.selectbox('Prior Therapy', ('Yes', 'No'))

    if st.button('Predict Survival Days'):
        st.write("Patient: ", 'Treatment: ', trt_selected, 'Cell: ', cell_selected, 'Karno: ', karno_selected,
                 'Diagnosis Time: ', diag_selected, 'Age: ', age_selected, 'Prior Therapy: ', prior_selected)
        st.write("Your algorithm: ", algorithm_selected)

        #create df with all inputs
        patient_df = pd.DataFrame({
            'trt': [trt_selected],
            'karno': [karno_selected],
            'diagtim': [diag_selected],
            'age': [age_selected],
            'prior': [prior_selected],
            'celltype_large': [],
            'celltype_smallcell': [],
            'celltype_squamous': []
        })

        #filling in the cell type matrix given selection
        patient_df = fill_celltype_matrix(patient_df, cell_selected)

        prediction = predict_suvival(patient_df, algorithm_selected)

        col1, col2 = st.columns([8, 6])

        if (prediction.flat[0] == 'ham'):
            with col1:
                st.markdown('### Ham! Looks like the message is good!')

            with col2:
                st.image('safe.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB",
                         output_format="auto")
        else:

            with col1:
                st.markdown('### Spam! Looks like spam. Be careful!')

            with col2:
                st.image('warning.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB",
                         output_format="auto")

elif choice == 'Data Exploration':
    st.title("Data Exploration")
    menu_list = ['Ham / Spam Word Freq by Size', 'Proportion of Ham vs Spam', 'Message length by Ham vs Spam']
    menu = st.radio("Menu", menu_list)

    if menu == 'Ham / Spam Word Freq by Size':
        st.header('Ham Word Frequency')
        st.image('Visualizations/ham freq - size.png')
        st.header('Spam Word Frequency')
        st.image('Visualizations/spam freq - size.png')

    elif menu == 'Proportion of Ham vs Spam':
        st.header('Pie Chart: Ham vs. Spam')
        st.image('Visualizations/pie chart.png')
        st.header('Bar Chart: Ham vs Spam')
        st.image('Visualizations/histgram.png')

    else:
        st.header('Ham vs. Spam Message length')
        st.image('Visualizations/ham spam msg len.png')

else:
    st.title("Model Evaluations")
    menu_list = ['SVC', 'Naive Bayes', 'KNN', 'Decision Tree', 'Logistic Regression', 'Random Forest']
    algorithm = st.selectbox("Algorithms", menu_list)

    classification_report = get_model_eval(algorithm)
    st.write(classification_report)