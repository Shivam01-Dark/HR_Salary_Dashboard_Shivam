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
