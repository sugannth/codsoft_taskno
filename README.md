# Spam SMS Detection using Machine Learning

## Introduction

Spam SMS messages are unwanted messages that often contain advertisements, scams, or fraudulent offers. This project uses Machine Learning techniques to automatically classify SMS messages as either **Spam** or **Ham (Not Spam)**.

## Objective

The objective of this project is to build a machine learning model capable of identifying spam messages accurately and helping users filter unwanted communications.

## Features

* Classifies SMS messages as Spam or Ham.
* Uses Natural Language Processing (NLP) techniques.
* Converts text data into numerical form using TF-IDF Vectorization.
* Trains a Machine Learning model for prediction.
* Evaluates model performance using accuracy and confusion matrix.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Dataset

The dataset contains SMS messages labeled as:

* Ham (Legitimate Message)
* Spam (Unwanted Message)

Example:

| Label | Message                                            |
| ----- | -------------------------------------------------- |
| Ham   | Hey, are you coming to class today?                |
| Spam  | Congratulations! You have won a free gift voucher. |

## Methodology

### 1. Data Collection

The SMS Spam Collection Dataset is loaded into Python.

### 2. Data Preprocessing

* Remove unnecessary columns
* Handle missing values
* Convert labels into numerical values

### 3. Feature Extraction

TF-IDF Vectorization is used to convert text messages into numerical feature vectors.

### 4. Model Training

The Multinomial Naive Bayes algorithm is trained using the processed dataset.

### 5. Model Evaluation

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1 Score

## Workflow

Dataset
→ Data Cleaning
→ Label Encoding
→ TF-IDF Vectorization
→ Train-Test Split
→ Model Training
→ Prediction
→ Evaluation

## Results

The trained model successfully classifies SMS messages as Spam or Ham with high accuracy and demonstrates the effectiveness of Machine Learning in text classification tasks.

## Future Enhancements

* Web Application Deployment
* Real-Time SMS Detection
* Deep Learning-Based Classification
* Support for Multiple Languages

## How to Run

1. Install required libraries:

pip install pandas numpy scikit-learn matplotlib seaborn

2. Run the project:

python spam_detector.py

## Conclusion

This project demonstrates how Machine Learning and Natural Language Processing can be combined to build an effective spam message detection system. The model helps automate SMS filtering and provides a practical application of text classification techniques.

## Author

Suganth K S

Developed as part of the CODSOFT Machine Learning Internship.
