import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv(r"D:\25CS270\Intern\spam.csv", encoding='latin-1')


# Check the dataset
print(df.head())
print(df.info())
print(df.shape)


#data cleaning
df = df[['v1', 'v2']]

df.rename(columns={
    'v1':'label',
    'v2':'message'
}, inplace=True)
#check missing values
print(df.isnull().sum())
#remove duplicates
df.drop_duplicates(inplace=True)

# Encode the labels
df['label'] = df['label'].map({
    'ham':0,
    'spam':1
})

# split features and target variable 
X = df['message']
y = df['label']

#convert the text to numbers 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size=0.2,
    random_state=42
)

# train the model,
model =MultinomialNB()
model.fit(X_train,y_train)

# make predictions
y_predict = model.predict(X_test)

# evaluate the model
#accuracy
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy:", accuracy)
#confusion matrix,
print("confusion matrix\n",confusion_matrix(y_test, y_predict))
#classification report
print("classification report\n",classification_report(y_test, y_predict))

#test the model with new messages
new_messages = [
    "Congratulations! You've won a free ticket to the PAris. Call now to claim your prize"]

new_vector= vectorizer.transform(new_messages)
prediction = model.predict(new_vector)
if prediction[0] == 1:
    print("The message is spam.")
else:
    print("The message is not spam,HAM.")




import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=df['label'])

plt.show()
