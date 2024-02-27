# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Generating a sample dataset
np.random.seed(0)
data = pd.DataFrame({
…
# Saving the model for future use
joblib.dump(model, 'accident_severity_model.pkl')

# Example of using the model to predict accident severity for hypothetical set of independent variables
example_data = np.array([[3, 2, 1, 2]])  # Assuming road condition=3, weather condition=2, time of day=1 (morning), type of vehicle=2
predicted_severity = model.predict(example_data)
print("Predicted Accident Severity:", predicted_severity[0])
