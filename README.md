# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Loading & Cleaning: The dataset is read, and irrelevant columns like sl_no and salary are dropped to focus on features useful for placement prediction.

2.Encoding Categorical Variables: All categorical columns are encoded into numeric values using LabelEncoder, making them suitable for machine learning models.

3.Feature and Target Selection: The independent variables (X) are extracted by dropping the target column status, which is used as the dependent variable (y).

4.Data Splitting and Scaling: The dataset is split into training and testing sets (80-20 split), and features are standardized using StandardScaler for improved model performance.

5.Model Training & Evaluation: A LogisticRegression model is trained on the scaled training data, and predictions are made on the test set. Model performance is evaluated using accuracy, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Rishab p doshi
RegisterNumber:  212224240134
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```
```
data = pd.read_csv("/content/Placement_Data.csv")
data.drop(['sl_no', 'salary'], axis=1, inplace=True)
```
```
label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
```
```
X = data.drop('status', axis=1)
y = data['status'] 
```
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
```
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
```
```
y_pred = model.predict(X_test_scaled)
y_pred
```
```
print("Accuracy:", accuracy_score(y_test, y_pred))

```
```
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```
```
print("Classification Report:\n", classification_report(y_test, y_pred))
```

## Output:
Predicted Values Of Y

![image](https://github.com/user-attachments/assets/fb575d36-3d21-42f6-84e0-ff3115a0a560)

Accuracy

![image](https://github.com/user-attachments/assets/4d1bceb2-6238-4348-b95f-82756f586da3)

Confusion matrix

![image](https://github.com/user-attachments/assets/2ba308d6-2906-4d4f-b7b3-0dc8601d3217)

Classifcation Matrix

![image](https://github.com/user-attachments/assets/00445486-ed49-4d28-b8c7-8857d34e5f2e)








## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
