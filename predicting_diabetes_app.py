
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os # Import os for file existence checks

# --- Project Title and Introduction ---
st.title('🩺 Diabetes Prediction App')
st.write('This interactive application leverages a machine learning model to predict the likelihood of diabetes based on several key health metrics. Input your data in the sidebar to get a real-time prediction!')

# --- Model Loading ---
# Loads the pre-trained machine learning model from 'diabetes_model.pkl'.
# This file is expected to be in the same directory as this script.
try:
    if os.path.exists('diabetes_model.pkl'):
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.sidebar.success("Machine learning model loaded successfully!")
    else:
        st.error("Error: Model file 'diabetes_model.pkl' not found. Please upload it to the Colab environment.")
        st.stop() # Halts app execution if the model is missing
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    st.stop() # Halts app execution for any other loading errors

# --- Sidebar for User Input Features ---
st.sidebar.header('Patient Input Features')
st.sidebar.write('Adjust the sliders below to input the patient\'s health metrics:')

# Function to collect user inputs via Streamlit sliders
def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3, help='Number of times pregnant.')
    glucose = st.sidebar.slider('Glucose (mg/dL)', 0, 200, 120, help='Plasma glucose concentration 2 hours in an oral glucose tolerance test.')
    blood_pressure = st.sidebar.slider('Blood Pressure (mmHg)', 0, 122, 70, help='Diastolic blood pressure.')
    skin_thickness = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 20, help='Triceps skin fold thickness.')
    insulin = st.sidebar.slider('Insulin (mu U/ml)', 0, 846, 79, help='2-Hour serum insulin.')
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0, help='Body Mass Index (weight in kg / (height in m)^2).')
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.471, help='A function that scores likelihood of diabetes based on family history.')
    age = st.sidebar.slider('Age (years)', 21, 81, 33, help='Age of the patient.')

    # Create a Pandas DataFrame from the collected inputs, as expected by the model.
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0]) # Single row DataFrame
    return features

# Get user inputs
df = user_input_features()

# Display user's input features
st.subheader('User Input Features')
st.dataframe(df)

# --- Prediction Section ---
st.subheader('Prediction Result')

# Button to trigger prediction
if st.button('Predict Diabetes'):
    try:
        # Perform prediction and get probability scores
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)

        st.markdown('---')

        # Display outcome
        if prediction[0] == 1:
            st.error('**Prediction: The patient is likely to have Diabetes.** 😔')
        else:
            st.success('**Prediction: The patient is likely NOT to have Diabetes.** 😊')

        # Display confidence levels
        st.write(f"Confidence (No Diabetes): **{prediction_proba[0][0]*100:.2f}%**")
        st.write(f"Confidence (Diabetes): **{prediction_proba[0][1]*100:.2f}%**")
        st.markdown('---')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check the input values or contact support if the issue persists.")

# --- Dataset Overview and Data Preprocessing Insights ---
st.subheader('Dataset Overview & Data Preprocessing Insights')
st.write("Understanding the characteristics of the dataset used for training the model and how missing values were handled:")

try:
    if os.path.exists('diabetes.csv'):
        dataset = pd.read_csv('diabetes.csv')

        st.write("### Dataset Dimensions")
        st.write(f"Number of Rows: {dataset.shape[0]}")
        st.write(f"Number of Columns: {dataset.shape[1]}")
        st.write(f"Initial Dimensions: (768, 9) - Including the 'Outcome' target variable.")
        st.write(f"Features for prediction: {dataset.shape[1] - 1} (excluding 'Outcome').")

        st.write("### Missing/Zero Values Analysis")
        st.write("For some features, a '0' value might represent a missing entry rather than a true zero (e.g., blood pressure cannot be 0).")

        features_with_potential_zeros_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

        null_counts = dataset.isnull().sum()
        zero_counts = (dataset[features_with_potential_zeros_as_missing] == 0).sum()

        stats_df = pd.DataFrame({
            'Null Count': null_counts,
            'Zero Count': pd.Series(zero_counts),
            'Total Count': dataset.shape[0]
        })
        stats_df['Null %'] = (stats_df['Null Count'] / stats_df['Total Count'] * 100).round(2)
        stats_df['Zero %'] = (stats_df['Zero Count'] / stats_df['Total Count'] * 100).round(2)

        stats_to_display = stats_df[['Null Count', 'Null %', 'Zero Count', 'Zero %']].fillna(0).astype({'Zero Count': int, 'Null Count': int})

        st.dataframe(stats_to_display)

        st.info("Note: For features like Glucose, BloodPressure, SkinThickness, Insulin, and BMI, a value of 0 often indicates a missing measurement in this dataset. These zeros were specifically handled during data preprocessing before model training.")

        st.markdown("""
        **Observations on Missing Data:**
        - Approximately 50% of the patients did not have their insulin levels measured. This initially raised a concern about potential data leakage, where doctors might only measure insulin levels in unhealthy-looking patients or after a preliminary diagnosis. If true, this could mean the model might not generalize well to data from doctors who measure insulin for every patient.
        - **Hypothesis Test:** To address this concern, it was checked whether the Insulin and SkinThickness features are correlated with the diagnostic outcome (healthy/diabetic).
        - **Conclusion:** The Insulin and SkinThickness measurements were found **not to be highly correlated with any given outcome**. As such, the concern of data leakage related to selective measurement could be ruled out.
        """)

        # --- Insulin Histogram (moved here for direct context) ---
        st.write("#### Insulin Distribution vs Outcome")
        try:
            if os.path.exists('Insulin_histogram.png'):
                st.image('Insulin_histogram.png', caption='This histogram illustrates the distribution of Insulin levels, separated by diabetes outcome (Blue = Healthy; Orange = Diabetes). It visually supports the conclusion that Insulin is not highly correlated with the outcome, alleviating data leakage concerns from selective measurement.', use_container_width=True)
            else:
                st.warning("Image 'Insulin_histogram.png' not found. Please ensure it's in the same directory.")
        except Exception as e:
            st.error(f"Error displaying Insulin histogram: {e}")

        # --- Skin Thickness Histogram (moved here for direct context) ---
        st.write("#### Skin Thickness Distribution vs Outcome")
        try:
            if os.path.exists('SkinThickness_histogram.png'):
                st.image('SkinThickness_histogram.png', caption='This histogram shows the distribution of Skin Thickness values, distinguished by diabetes outcome (Blue = Healthy; Orange = Diabetes). Similar to Insulin, it helps in understanding the lack of strong correlation with the outcome despite zero values.', use_container_width=True)
            else:
                st.warning("Image 'SkinThickness_histogram.png' not found. Please ensure it's in the same directory.")
        except Exception as e:
            st.error(f"Error displaying SkinThickness histogram: {e}")

        st.markdown("""
        **Handling Erroneous Zero Values:**
        - Despite ruling out data leakage from selective measurement, the zero values in categories like Insulin and SkinThickness are still erroneous (e.g., a person cannot have 0 skin thickness). These values should not be included directly in the model.
        - **Imputation Strategy:** It is best practice to replace these erroneous zero values with some distribution of values, typically near the median measurement of that feature.
        - **Preventing Data Leakage during Imputation:** It is crucial to impute these values *after* the `train_test_split` function has been applied during the model training phase. This prevents another form of data leakage, ensuring that information from the testing data (e.g., its median value) is not used when calculating the imputation values for the training data. The original notebook's data preprocessing steps confirm that null values were indeed replaced with median values.

        Because all erroneous, missing, and null values were replaced with median values during the data preprocessing stage of the notebook, the data was then ready for model training and evaluation.
        """)

    else:
        st.warning("Dataset file 'diabetes.csv' not found for Dataset Overview. Please upload it to the Colab environment.")

except Exception as e:
    st.error(f"An error occurred while loading or processing dataset for overview: {e}")


# --- General Exploratory Data Analysis Visualizations ---
# This section now focuses on general EDA plots, like the correlation heatmap.
st.subheader('Exploratory Data Analysis: Key Visualizations')
st.write("Below are pre-generated plots providing further insights into the dataset's features and their relationships.")

# Display Correlation Heatmap
st.write("### Feature Correlation Heatmap")
try:
    if os.path.exists('Correlation_Heatmap.png'):
        st.image('Correlation_Heatmap.png', caption='This heatmap visualizes the Pearson correlation coefficients between all numerical features in the dataset. Darker colors (closer to 1 or -1) indicate stronger linear relationships, while lighter colors (closer to 0) suggest weaker ones. It helps identify multicollinearity and important feature relationships.', use_container_width=True)
    else:
        st.warning("Image 'Correlation_Heatmap.png' not found. Please ensure it's in the same directory.")
except Exception as e:
    st.error(f"Error displaying Correlation Heatmap: {e}")


# --- Algorithm Performance & Feature Importance ---
st.subheader('Algorithm Performance & Feature Importance')
st.write(
    """
    To ensure the selection of the most suitable machine learning model for diabetes prediction,
    several common classification algorithms were rigorously evaluated. This was primarily done
    using K-Fold Cross-Validation, a robust technique that assesses how well models generalize
    to unseen data by partitioning the dataset into multiple subsets for training and testing.
    """
)

st.write("### K-Fold Cross-Validation Accuracy Scores:")
st.write(
    """
    The accuracy scores presented below represent the average performance of each algorithm
    across multiple cross-validation folds. The accompanying standard deviation (e.g., +/- 0.0732)
    quantifies the variability of the model's performance across these folds. A higher accuracy
    score combined with a lower standard deviation generally indicates a more consistent and
    reliable model that generalizes well.
    """
)
st.code("""
Nearest Neighbors: 0.5830 (+/- 0.0732)
Linear SVM: 0.6270 (+/- 0.0389)
RBF SVM: 0.6515 (+/- 0.0043)
Gaussian Process: 0.6238 (+/- 0.0835)
Decision Tree: 0.5391 (+/- 0.0435)
Random Forest: 0.6091 (+/- 0.0336)
MLPClassifier: 0.6042 (+/- 0.0396)
AdaBoost: 0.6123 (+/- 0.0510)
Naive Bayes: 0.6091 (+/- 0.0529)
QDA: 0.5783 (+/- 0.1088)
""")

st.write("### Visual Comparison of Algorithm Performance (Box Plot)")
try:
    # Changed filename to match common practice for boxplots
    if os.path.exists('algorithm_box_and_whisker.png'):
        st.image('algorithm_box_and_whisker.png', caption='This boxplot visually compares the distribution of cross-validation accuracy scores for each machine learning algorithm. The box represents the interquartile range (IQR), the central line is the median accuracy, and the whiskers extend to show the overall range of performance. Outliers are plotted as individual points.', use_container_width=True)
    else:
        st.warning("Image 'algorithm_box_and_whisker.png' not found. Please ensure it's in the same directory.")
except Exception as e:
    st.error(f"Error displaying Algorithm Comparison Box Plot: {e}")


st.write("### Learning Curve for Decision Tree Classifier")
st.write(
    """
    A learning curve is a diagnostic tool that helps analyze a model's bias and variance.
    It plots the model's performance on both the training dataset and a separate cross-validation
    set as a function of the increasing number of training examples.
    """
)
try:
    if os.path.exists('learning curve.png'):
        st.image('learning curve.png', caption='This learning curve specifically for a Decision Tree Classifier shows training (red) and cross-validation (green) scores. A significant gap between the lines suggests high variance (overfitting), while consistently low scores on both indicate high bias (underfitting).', use_container_width=True)
    else:
        st.warning("Image 'learning curve.png' not found. Please ensure it's in the same directory.")
except Exception as e:
    st.error(f"Error displaying Learning Curve Plot: {e}")

st.write("### Feature Importance by Classifier")
st.write(
    """
    Feature importance indicates how much each feature contributed to the model's prediction. A higher absolute coefficient or feature importance score implies greater influence.
    It's noteworthy that many kernel authors neglected to deal with the null values (specifically zero values) discussed earlier. This oversight, however, did not significantly impact the performance of most of their models. This is primarily because, as observed through feature importance analysis, Insulin and SkinThickness measurements are actually very poor predictors and are consistently assigned low feature importances compared to more influential features like blood glucose levels and body mass index.
    """
)

st.write("#### DecisionTreeClassifier - Feature Importance:")
st.code("""
                    Variable  absCoefficient
1                   Glucose        0.645256
5                       BMI        0.247421
7                       Age        0.107322
0               Pregnancies        0.000000
2             BloodPressure        0.000000
3             SkinThickness        0.000000
4                   Insulin        0.000000
6  DiabetesPedigreeFunction        0.000000
""")
st.write(f"Accuracy of DecisionTreeClassifier: 0.78")


st.write("#### RandomForestClassifier - Feature Importance:")
st.code("""
                    Variable  absCoefficient
1                   Glucose        0.430867
4                   Insulin        0.143902
6  DiabetesPedigreeFunction        0.126688
7                       Age        0.112554
5                       BMI        0.100692
0               Pregnancies        0.056478
2             BloodPressure        0.019575
3             SkinThickness        0.009244
""")
st.write(f"Accuracy of RandomForestClassifier: 0.78")


st.write("#### XGBClassifier - Feature Importance:")
st.code("""
                    Variable  absCoefficient
5                       BMI        0.201087
6  DiabetesPedigreeFunction        0.190217
1                   Glucose        0.184783
7                       Age        0.114130
4                   Insulin        0.112319
0               Pregnancies        0.074275
2             BloodPressure        0.067029
3             SkinThickness        0.056159
""")
st.write(f"Accuracy of XGBClassifier: 0.82")

st.markdown("""
**Overall Conclusion:**
In summary, we were able to predict diabetes from medical records with an accuracy of approximately **82%**. This was achieved by utilizing tree-based classifiers, which effectively focus on the most important features such as blood glucose levels and body mass index. Interestingly, the model's performance remains robust even with fewer features; we observe only a **5% reduction in accuracy** when considering only blood glucose levels and body mass index, highlighting their dominant predictive power.
""")


# --- Footer and Disclaimer ---
st.sidebar.markdown('---')
st.sidebar.info('This app utilizes a machine learning model to predict diabetes risk. It is intended solely for informational and educational purposes and should not be used as a substitute for professional medical advice, diagnosis, or treatment.')
