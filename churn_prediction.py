import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
# Load the dataset
df = pd.read_csv(r"D:\25CS270\Intern\Task-3 customor churn perdiction\Churn_Modelling.csv")

print(df.head())
print(df.shape)

# Preprocess the data
print(df.info())
#check missing values
print(df.isnull().sum())
#remove unnecessary columns
df.drop(
    ["RowNumber","CustomerId","Surname"],
    axis=1,
    inplace=True
)

#convert text columns to numbers

le_geo = LabelEncoder()

df["Geography"] = le_geo.fit_transform(
    df["Geography"]
)

le_gender = LabelEncoder()

df["Gender"] = le_gender.fit_transform(
    df["Gender"]
)

#spliting into features and target variable
y = df["Exited"]
X = df.drop("Exited", axis=1)
#spliting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

#predicting the test set results

y_pred = model.predict(X_test)

#checking the accuracy of the model
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)

#classification report 
print(
    classification_report(
        y_test,
        y_pred
    )
)

#confusion matrix 
cm = confusion_matrix(
    y_test,
    y_pred
)

print("the confusion matrix is:\n", cm)

#test a new customer data
customer = pd.DataFrame({
    'CreditScore':[600],
    'Geography':[1],
    'Gender':[1],
    'Age':[40],
    'Tenure':[5],
    'Balance':[50000],
    'NumOfProducts':[2],
    'HasCrCard':[1],
    'IsActiveMember':[1],
    'EstimatedSalary':[60000]
})

prediction = model.predict(customer)

if prediction[0] == 1:
    print("Customer Will Churn")
else:
    print("Customer Will Stay")


#visualization of feature importance
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=df["Exited"])

plt.title("Customer Churn Distribution")
plt.show()
