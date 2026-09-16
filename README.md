# 🏦 Loan Prediction System

A machine learning-based Loan Approval Prediction System that predicts whether a loan application is likely to be approved or rejected based on applicant and campaign-related information.

The project includes a trained machine learning pipeline and an interactive Streamlit web application for making predictions.

## 🚀 Project Overview

This project demonstrates an end-to-end machine learning workflow:

- Data preprocessing and feature preparation
- Feature encoding and scaling
- Model training and evaluation
- Hyperparameter tuning
- Saving the trained model as a `.pkl` file
- Building an interactive Streamlit application
- Making real-time predictions through a web interface

## 🧠 Machine Learning Model

The project uses a **Tuned Random Forest Classifier** for prediction.

The saved model is stored as:

`best_pipeline.pkl`

The pipeline handles the required preprocessing before passing the data to the trained model.

## 📊 Input Features

The Streamlit application provides user-friendly inputs including:

- Age
- Job
- Marital Status
- Education
- Credit Default
- Housing Loan
- Personal Loan
- Contact Type
- Contact Month
- Day of Week
- Number of Contacts in Current Campaign
- Previous Contacts
- Days Since Previous Contact
- Previous Campaign Outcome

The application converts these inputs into the feature representation expected by the trained model.

## 🌐 Streamlit Application

The application provides an interactive interface where users can enter applicant information and receive:

- Loan approval prediction
- Approval probability
- Prediction result through a simple web interface

The application is implemented in:

`app.py`

## 📁 Project Structure

```text
Loan-Prediction-System/
│
├── app.py
├── best_pipeline.pkl
├── loan_detection.csv
├── loan_detection_proj (1).ipynb
├── requirements.txt
└── README.md
