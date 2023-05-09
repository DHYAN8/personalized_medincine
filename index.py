import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk

# load the dataset
df = pd.read_csv('drug200.csv')

# replace categorical values with numerical values
dataset = df.replace({'F': 1, 'M': 0})
dataset1 = dataset.replace({'LOW': -1,'NORMAL':0 ,'HIGH': 1})

# split the dataset into input and output variables
X = dataset1.drop('Drug', axis=1)
y = dataset1['Drug']

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train a logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# define a function to predict the drug type for a patient based on their input data
def predict_drug(age, sex, bp, cholesterol, na_to_k_ratio):
    # create a pandas dataframe with the patient data
    user_input = pd.DataFrame({'Age': [age], 'Sex': [sex], 'BP': [bp], 'Cholesterol': [cholesterol], 'Na_to_K': [na_to_k_ratio]})
    # make a prediction using the logistic regression model
    prediction = lr.predict(user_input)
    # return the predicted drug type
    return prediction[0]

# define a function to handle button click events
def predict_drug():
    # get the input values from the user interface
    age = float(age_entry.get())
    sex = sex_var.get()
    bp = bp_var.get()
    cholesterol = cholesterol_var.get()
    na_to_k_ratio = float(na_to_k_entry.get())
    
    # convert sex input from string to numerical value
    if sex == "M":
        sex = 0
    else:
        sex = 1
        
    # convert blood pressure input from string to numerical value
    if bp == "LOW":
        bp = -1
    elif bp == "NORMAL":
        bp = 0
    else:
        bp = 1
    
    # convert cholesterol input from string to numerical value
    if cholesterol == "LOW":
        cholesterol = -1
    elif cholesterol == "NORMAL":
        cholesterol = 0
    else:
        cholesterol = 1
    
    # create a pandas dataframe with the patient data
    user_input = pd.DataFrame({'Age': [age], 'Sex': [sex], 'BP': [bp], 'Cholesterol': [cholesterol], 'Na_to_K': [na_to_k_ratio]})
    # make a prediction using the logistic regression model
    prediction = lr.predict(user_input)
    # display the predicted drug type
    prediction_label.configure(text=f"The predicted drug type for the patient is: {prediction[0]}")

# create the main window
root = tk.Tk()
root.title(" Disease  drug Type Prediction")

# set the font and background color of the window
root.configure(bg='white')
root.option_add('*Font', ('Verdana', 10))

# create a label for the age input
age_label = tk.Label(root, text="Age:", justify='left')
age_label.grid(row=0, column=0)

# create an entry box for the age input
age_entry = tk.Entry(root)
age_entry.grid(row=0, column=1,)

sex_label = tk.Label(root, text="Sex:", justify='left')
sex_label.grid(row=1, column=0)

sex_var = tk.StringVar()
sex_var.set("F")
sex_male = tk.Radiobutton(root, text="Male", variable=sex_var, value="M")
sex_female = tk.Radiobutton(root, text="Female", variable=sex_var, value="F")
sex_male.grid(row=1, column=1)
sex_female.grid(row=1, column=2)

bp_label = tk.Label(root, text="Blood Pressure:", justify='left')
bp_label.grid(row=2, column=0)

bp_var = tk.StringVar()
bp_var.set("LOW")
bp_choices = ["LOW", "NORMAL", "HIGH"]
bp_dropdown = tk.OptionMenu(root, bp_var, *bp_choices)
bp_dropdown.grid(row=2, column=1)

cholesterol_label = tk.Label(root, text="Cholesterol:", justify='left')
cholesterol_label.grid(row=3, column=0)

cholesterol_var = tk.StringVar()
cholesterol_var.set("LOW")
cholesterol_choices = ["LOW", "NORMAL", "HIGH"]
cholesterol_dropdown = tk.OptionMenu(root, cholesterol_var, *cholesterol_choices)
cholesterol_dropdown.grid(row=3, column=1)

na_to_k_label = tk.Label(root, text="Na to K Ratio:", justify='left')
na_to_k_label.grid(row=4, column=0)

na_to_k_entry = tk.Entry(root)
na_to_k_entry.grid(row=4, column=1)

predict_button = tk.Button(root, text="Predict", command=predict_drug)
predict_button.grid(row=5, column=1)

prediction_label = tk.Label(root, text="", justify='center')
prediction_label.grid(row=6, column=0, columnspan=3)

root.mainloop()