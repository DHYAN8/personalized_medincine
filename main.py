import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('drug200.csv')
dataset = df.replace({'F': 1, 'M': 0})
dataset1 = dataset.replace({'LOW': -1,'NORMAL':0 ,'HIGH': 1})

X = dataset1.drop('Drug', axis=1)
y = dataset1['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)



print("Please enter the patient details for drug prediction:")
age = float(input("Age: "))
sex = input("Sex (M/F): ")
if sex == "F":
  sex = 1
else:
  sex = 0
bp = input("Blood Pressure (LOW/NORMAL/HIGH): ")
if bp == "LOW":
  bp = -1
elif bp == "NORMAL":
  bp = 0
else:
  bp = 1
cholesterol = input("Cholesterol (LOW/NORMAL/HIGH): ")
if cholesterol == "LOW":
  cholesterol = -1
elif cholesterol == "NORMAL":
  cholesterol = 0
else:
  cholesterol = 1
na_to_k_ratio = float(input("Na to K Ratio: "))

user_input = pd.DataFrame({'Age': [age], 'Sex': [sex], 'BP': [bp], 'Cholesterol': [cholesterol], 'Na_to_K': [na_to_k_ratio]})


prediction = lr.predict(user_input)

print("The predicted drug type for the patient is:", prediction)