import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

conn = sqlite3.connect(':memory:')

senate_df = pd.read_csv(r'C:\Users\Work\Desktop\Are they there\Senate&HOR - Age Senate.csv')
hor_df = pd.read_csv(r'C:\Users\Work\Desktop\Are they there\Senate&HOR - Age HOR.csv')

senate_df.to_sql('senate', conn, index=False)
hor_df.to_sql('hor', conn, index=False)

sql_query = """
SELECT Name, Age FROM senate
UNION
SELECT Name, Age FROM hor
"""
data = pd.read_sql_query(sql_query, conn)

data['Dementia_Label'] = (data['Age'] >= 65).astype(int)

common_age = data['Age'].mode()[0]
avg_age = data['Age'].mean()
youngest_people = data[data['Age'] == data['Age'].min()]
oldest_people = data[data['Age'] == data['Age'].max()]

print(f"Most common age: {common_age}")
print(f"Average age: {avg_age:.2f}")
print("Youngest person:")
print(youngest_people[['Name', 'Age']])
print("Oldest person:")
print(oldest_people[['Name', 'Age']])

# Logistic Regression
data['Age_Above_65'] = (data['Age'] >= 65).astype(int)
X = data[['Age_Above_65']]
y = data['Dementia_Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plotting
plt.figure(figsize=(8, 6))
values, bins, bars = plt.hist(data['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)

for value, bar in zip(values, bars):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f'{int(value)}', ha='center', va='bottom')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Test Data')
plt.plot(X_test, log_reg.predict_proba(X_test)[:, 1], color='red', label='Logistic Regression')
plt.title('Logistic Regression Predictions')
plt.xlabel('Age Above 65')
plt.ylabel('Probability of Dementia')
plt.legend()

percentage_positive_predictions = (sum(y_pred == 1) / len(y_pred)) * 100
print(f"Percentage of people who may have dementia: {percentage_positive_predictions:.2f}%")
