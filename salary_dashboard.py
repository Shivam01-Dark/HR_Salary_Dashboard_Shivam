import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("datasets/HRDataset_v14.csv")

# Display first 5 rows
print(data.head())

# Salary distribution visualization
plt.hist(data['Salary'])
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.show()

# Average salary by Position
avg_salary = data.groupby('Position')['Salary'].mean()
avg_salary.plot(kind='bar')
plt.title("Average Salary by Position")
plt.xlabel("Position")
plt.ylabel("Average Salary")
plt.show()

# ---------- Linear Regression Model ----------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Select only numeric columns
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
X = data[numeric_cols[:-1]]  # all numeric columns except last column as features
y = data[numeric_cols[-1]]   # last numeric column as target (Salary expected)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
print("\n--- Linear Regression Results ---")
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Actual vs Predicted Visualization
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted Salary")
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.show()
