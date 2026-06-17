# Credit Card Fraud Detection using Machine Learning

## Introduction

Credit card fraud is one of the major challenges faced by financial institutions. This project uses Machine Learning techniques to identify fraudulent transactions based on transaction details.

## Objective

To build a machine learning model that classifies credit card transactions as legitimate or fraudulent.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Dataset

The dataset contains more than 1.2 million transaction records with various transaction-related features.

Selected Features:

* Transaction Amount
* Latitude
* Longitude
* City Population
* Unix Time
* Merchant Latitude
* Merchant Longitude

Target Variable:

* is_fraud

  * 0 : Legitimate Transaction
  * 1 : Fraudulent Transaction

## Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Selection
4. Train-Test Split
5. Train Decision Tree Classifier
6. Predict Transactions
7. Evaluate Model Performance

## Algorithm Used

Decision Tree Classifier

## Evaluation Metrics

* Accuracy Score
* Classification Report
* Confusion Matrix

## Results

The model achieved an accuracy of approximately 99.28% and successfully classified fraudulent and legitimate transactions.

## Future Enhancements

* Handle Imbalanced Dataset
* Use Random Forest
* Use SMOTE for Better Fraud Detection
* Deploy as Web Application

## Conclusion

This project demonstrates how Machine Learning can be used to detect fraudulent financial transactions and improve the security of digital payment systems.

## Author

Developed as part of the CODSOFT Machine Learning Internship.
