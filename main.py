import pandas as pd

from NILValueModel import st, predict_suvival

menu = ["Home", "Data Exploration", "Prediction: Survival Days"]
choice = st.sidebar.selectbox('Navigation', menu)
st.sidebar.markdown("Please use drop down to navigate to different pages")


if choice == 'Home':
    st.title('CSDA 1040 Group 2:')
    st.title('Survival Prediction Model')
    st.header(
        "In this project, our model predicts how many days a given veteran lung cancer patient will survive." +
        "Our data is built on a study done for veterans with lung cancer and their survival." +
        "We are using the veteran lung data from")
    st.subheader("http://www-eio.upc.edu/~pau/cms/rdata/doc/survival/veteran.html")

    st.subheader("Lakshmi Sameera Vemula")
    st.subheader("Cheng Qian")
    st.subheader("Joey Santiago")
    st.subheader("Nareshini Dookhy")
    st.subheader("Fatima Anam Ahmad")



elif choice == 'Prediction: Survival Days':

    st.title('Survival Days Prediction for Lung Cancer')

    algorithm_selected=st.selectbox("Select an algorithm:", ['Cox Proportional-Hazards Model', 'Weibull Accelerated Failure Time Model'])

    #description
    st.write("The survival days is predicted based on the following attributes of a patient:" +
             "  \n  \t  1. Treatment - Standard or Test" +
             "  \n  \t  2. Cancer Cell Type - Squamous, Smallcell, Adeno, Large" +
             "  \n  \t  3. Karnofsky Performance Score (100=good)" +
             "  \n  \t  4. Diagnosis Time - Months from Diagnosis to Randomisation" +
             "  \n  \t  5. Age" +
             "  \n  \t  6. If they've had Prior Therapy")

    #inputs
    trt_selected = st.selectbox( 'Treatment Type', ('Standard', 'Test'))
    cell_selected = st.selectbox('Cancer Cell Type', ('Squamous', 'Smallcell', 'Adeno', 'Large'))
    karno_selected = st.number_input('Karnofsky Score', min_value=1, max_value=100, value=50, step=1)
    diag_selected = st.number_input('Diagnosis Time', min_value=1, max_value=100, value=50, step=1)
    age_selected = st.number_input('Age', min_value=30, max_value=100, value=65, step=1)
    prior_selected = st.selectbox('Prior Therapy', ('Yes', 'No'))

    if st.button('Predict Survival Days'):
        st.write('Treatment: ', trt_selected, 'Cell: ', cell_selected, 'Karno: ', karno_selected,
                 'Diagnosis Time: ', diag_selected, 'Age: ', age_selected, 'Prior Therapy: ', prior_selected)
        st.write("Your algorithm: ", algorithm_selected)

        #create df with all inputs
        patient_df = pd.DataFrame({
            'trt': [trt_selected],
            'karno': [karno_selected],
            'diagtime': [diag_selected],
            'age': [age_selected],
            'prior': [prior_selected],
            'celltype_large': [0],
            'celltype_smallcell': [0],
            'celltype_squamous': [0]
        })

        prediction = predict_suvival(patient_df, cell_selected,algorithm_selected)

        col1, col2 = st.columns([8, 6])

        st.write('The patient is predicted to survive ', round(prediction.at[0]), ' days')

else:
    st.title("Data Exploration")
    menu_list = ['Observed Deaths vs Censored', 'Feature Density', 'Suvival Function',
                 'Survival By Treatment', 'Survival By Cell Type']
    menu = st.radio("Menu", menu_list)

    if menu == 'Observed Deaths vs Censored':
        st.header('Observed Deaths (1) vs Alive (0) at the end of Study')
        st.write('The majority of samples are not censored so prediction should not be hindered.')
        st.image('Visualizations/dead-v-alive.png')

    elif menu == 'Feature Density':
        st.header('Age')
        st.image('Visualizations/feat-age.png')
        st.header('Cell Type')
        st.image('Visualizations/feat-celltype.png')
        st.header('Diagnosis Time')
        st.image('Visualizations/feat-diagtime.png')
        st.header('Prior Therapy')
        st.image('Visualizations/feat-prior.png')
        st.header('Treatment')
        st.image('Visualizations/feat-trt.png')

    elif menu == 'Survival By Treatment':
        st.header('Survival Function By Treatment (Kaplan Meier)')
        st.write('There is not a pronounced difference in the standard survival curve vs test survival curve. ' +
                 'We cannot conclude that either has better survival chances than the other.')
        st.image('Visualizations/survival-by-trt.png')

    elif menu == 'Suvival Function':
        st.header('Overall Survival Function (Kaplan Meier)')
        st.image('Visualizations/survival-fn.png')

    else:
        st.header('Survival Function By Cancer Cell Type (Kaplan Meier)')
        st.write('There is a noticeable difference between two groups. Patients with squamous or large cells ' +
                 'seem to have a better prognosis compared to patients with small or adeno cells.')
        st.image('Visualizations/survival-by-cell.png')
