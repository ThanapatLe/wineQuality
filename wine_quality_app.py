import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import streamlit as st

# Read the CSV file into a DataFrame
df = pd.read_csv('winequality-red.csv')

# Define the features and target variable
features = ['density', 'alcohol', 'pH']
target = 'quality'

# Split the data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'wine_quality_model.pkl')

# Load the model from a file (if needed)
loaded_model = joblib.load('wine_quality_model.pkl')

# Streamlit app
st.title("Wine Quality Prediction App")

# Input for user
st.sidebar.header("Input Features")
density = st.sidebar.number_input("Density", min_value=0.99, max_value=1.04, value=0.9956)
alcohol = st.sidebar.number_input("Alcohol", min_value=8.0, max_value=15.0, value=10.4)
ph = st.sidebar.number_input("pH", min_value=2.5, max_value=4.0, value=3.23)

# Make a prediction
sample_input = [[density, alcohol, ph]]
predicted_quality = loaded_model.predict(sample_input)

# Display the prediction
st.write(f"Predicted Quality: {predicted_quality[0]:.2f}")

# Make predictions on the test data
y_pred = loaded_model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate R-squared (R2)
r2 = r2_score(y_test, y_pred)

# Display MSE and R2
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R2): {r2:.2f}")
