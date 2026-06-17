

import pandas as pd

df = pd.read_csv(r"D:/25CS270/Intern/Task-4 Credit card fraud detection/fraudTrain.csv")

print(df.head())
print(df.shape)
print(df.info())

#check fraudulent transactions
print(df[df["is_fraud"] == 1].value_counts())

#keep only necessary columns
df = df[
[
    "amt",
    "lat",
    "long",
    "city_pop",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud"
]
]
print(df.head())

# feature and target variable
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]


# spliting into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the model
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)

#prediction
y_pred = model.predict(X_test)

#checking the accuracy of the model
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("Accuracy:", accuracy)

#classification report
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred
    )
)

#confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")
print(cm)

#test a new transaction data
sample_transaction = pd.DataFrame({
    "amt": [5000],
    "lat": [35.0],
    "long": [-80.0],
    "city_pop": [100000],
    "unix_time": [1325376018],
    "merch_lat": [36.0],
    "merch_long": [-81.0]
})

prediction = model.predict(sample_transaction)

if prediction[0] == 1:
    print("Fraud Transaction")
else:
    print("Legitimate Transaction")


#visualization of feature importance
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=df["is_fraud"])

plt.title("Fraud Distribution")
plt.show()


#confusion matric graph
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()
