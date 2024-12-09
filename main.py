import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Title of the Streamlit App
st.title("Salary Analysis and Prediction")

# Load dataset directly from the file in your directory
file_path = "file/Salary Data.xlsx"
try:
    # Load dataset
    data = pd.read_excel(file_path)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    st.write("### Data Exploration and Preparation")

    # Descriptive Statistics
    st.write("**Basic Descriptive Statistics**")
    st.write(data.describe(include='all'))

    # Ensure column headers match the report
    expected_columns = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Salary"]
    if all(col in data.columns for col in expected_columns):
        # Handle missing values
        st.write("### Data Preprocessing")
        missing_values = data.isnull().sum()
        st.write("Missing values per column:", missing_values)

        # Fill or drop missing values (example: filling with median or mode)
        data = data.fillna(data.median(numeric_only=True)).fillna(data.mode().iloc[0])
        st.write("Missing values handled.")

        # Encode categorical variables
        label_encoders = {}
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

        st.write("Categorical columns encoded.")

        # Visualize Salary Distribution
        st.write("### Salary Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data['Salary'], kde=True, ax=ax)
        ax.set_title("Salary Distribution")
        st.pyplot(fig)

        # Correlation Heatmap
        st.write("### Correlation Heatmap")
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Select features and target
        st.write("### Feature Selection")
        target_column = "Salary"
        feature_columns = st.multiselect(
            "Select feature columns", 
            [col for col in data.columns if col != target_column],
            default=["Age", "Gender", "Education Level", "Years of Experience"]
        )

        if feature_columns:
            X = data[feature_columns]
            y = data[target_column]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Ridge Regression
            st.write("### Ridge Regression")
            alpha = st.slider("Select alpha value for Ridge Regression", min_value=0.1, max_value=10.0, step=0.1)
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)

            # Predictions and Evaluation
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write("Mean Squared Error:", mse)
            st.write("R-Squared Score:", r2)

            # Display coefficients
            st.write("### Feature Coefficients")
            coefficients = pd.DataFrame({"Feature": feature_columns, "Coefficient": model.coef_})
            st.dataframe(coefficients)
    else:
        st.error("The dataset does not have the required columns: " + ", ".join(expected_columns))
except FileNotFoundError:
    st.error("The specified file was not found. Please check the file path.")

st.write("### Insights")
st.write("- Ridge Regression indicates the relative importance of features in predicting salaries.")
st.write("- Higher coefficients suggest greater influence of a feature on salary.")
st.write("- Use these insights to understand trends like how education, experience, or specific job roles impact salary.")