from django.shortcuts import render,redirect
from Mainapp.models import*
from Userapp.models import*
from Adminapp.models import *
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse
import os
import shutil
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from django.contrib import messages
from keras.models import load_model
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from .models import SVM
import joblib  # Assuming 'SVM','KNN' is your model in models.py to store the metrics

#gradient boost machine algo for getting acc ,precession , recall , f1 score
# Create your views here.
def adminlogout(req):
    messages.info(req,'You are logged out...!')
    return redirect('index')

def admindashboard(req):
    return render(req,'admin/admin-dashboard.html')


def SVM_alg(req):
  return render(req,'admin/SVM_alg.html')

def NB_alg(req):
  return render(req,'admin/NB_alg.html')

def LR_alg(req):
  return render(req,'admin/LR_alg.html')



def admingraph(req):
    # Fetch the latest r2_score for each model
    svm_details1 = SVM.objects.last()
    SVM1 = svm_details1.accuracy

    lr_details2 = LR.objects.last()
    LR1 = lr_details2.accuracy

    nb_details2 = NB.objects.last()
    NB1 = nb_details2.accuracy


    print('SVM1','NB1','LR1')
    print(SVM1,NB1,LR1)
    return render(req, 'admin/admin-graph-analysis.html',{'SVMl':SVM1,'LR1':LR1,'NB1':NB1})


# views.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
from django.shortcuts import render
from .models import SVM  # Assuming a model to store results
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def SVM_btn(request):
    # Load the dataset
    df = pd.read_csv('D:/Ransomware_Detect2/Ransomware_Detect2/dataset/data_file.csv')

    # Check the distribution of the 'Benign' column
    print("Class distribution in target variable (Benign):")
    print(df['Benign'].value_counts())

    # If only one class is present, raise an error
    if df['Benign'].nunique() == 1:
        raise ValueError("The target variable 'Benign' must have both 0 and 1 values. Currently, it has only one class.")

    # Separate features (X) and target variable (y)
    X = df.drop(columns=['Benign', 'FileName', 'md5Hash'])  # Drop unnecessary columns (if any)
    y = df['Benign']

    # Split the data into train and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature scaling for better performance with SVM
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Create and train the SVM model with a linear kernel
    svm_model = SVC(kernel='linear', C=1, random_state=42)
    svm_model.fit(X_res, y_res)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation metrics
    print("\nSVM Model Evaluation")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


    # Save the SVM model
    joblib.dump(svm_model, 'svm_model.pkl')

    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    

    Name="SVM Model"
    # Save the metrics in the database (Django model)
    SVM.objects.create(
        name=Name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1
    )

    # Fetch the latest result to pass to the template
    results = SVM.objects.last()

    # Prepare the results for rendering
    context = {
        'name': results.name,
        'accuracy': round(results.accuracy, 2),
        'precision': round(results.precision, 2),
        'recall': round(results.recall, 2),
        'f1_score': round(results.f1_score, 2)
    }

    # Render the results to the template
    return render(request, 'admin/SVM_alg.html', context)


# views.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
from django.shortcuts import render
from .models import NB  # Assuming a model to store results
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def NB_btn(request):
    # Load the dataset
    df = pd.read_csv('D:/Ransomware_Detect2/Ransomware_Detect2/dataset/data_file.csv')

    # Check the distribution of the 'Benign' column
    print("Class distribution in target variable (Benign):")
    print(df['Benign'].value_counts())

    # If only one class is present, raise an error
    if df['Benign'].nunique() == 1:
        raise ValueError("The target variable 'Benign' must have both 0 and 1 values. Currently, it has only one class.")

    # Separate features (X) and target variable (y)
    X = df.drop(columns=['Benign', 'FileName', 'md5Hash'])  # Drop unnecessary columns (if any)
    y = df['Benign']

    # Split the data into train and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature scaling for better performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Create and train the Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_res, y_res)

    # Make predictions on the test set
    y_pred = nb_model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation metrics
    print("\nNaive Bayes Model Evaluation")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the Naive Bayes model to a pickle file
    joblib.dump(nb_model, 'naive_bayes_model.pkl')

    # Save the scaler to a pickle file
    joblib.dump(scaler, 'scaler.pkl')

    # Save the metrics in the database (Django model)
    NB.objects.create(
        name="Naive Bayes Model",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1
    )

    # Fetch the latest result to pass to the template
    results = NB.objects.last()

    # Prepare the results for rendering
    context = {
        'name': results.name,
        'accuracy': round(results.accuracy, 2),
        'precision': round(results.precision, 2),
        'recall': round(results.recall, 2),
        'f1_score': round(results.f1_score, 2)
    }

    # Render the results to the template
    return render(request, 'admin/NB_alg.html', context)


# views.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
from django.shortcuts import render
from .models import LR  # Assuming a model to store results
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def LR_btn(request):
    # Load the dataset
    df = pd.read_csv('D:/Ransomware_Detect2/Ransomware_Detect2/dataset/data_file.csv')

    # Check the distribution of the 'Benign' column
    print("Class distribution in target variable (Benign):")
    print(df['Benign'].value_counts())

    # If only one class is present, raise an error
    if df['Benign'].nunique() == 1:
        raise ValueError("The target variable 'Benign' must have both 0 and 1 values. Currently, it has only one class.")

    # Separate features (X) and target variable (y)
    X = df.drop(columns=['Benign', 'FileName', 'md5Hash'])  # Drop unnecessary columns (if any)
    y = df['Benign']

    # Split the data into train and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature scaling for better performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Create and train the Logistic Regression model
    linear_model = LogisticRegression(random_state=42)
    linear_model.fit(X_res, y_res)

    # Make predictions on the test set
    y_pred = linear_model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation metrics
    print("\nLogistic Regression Model Evaluation")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the Logistic Regression model to a pickle file
    joblib.dump(linear_model, 'linear_model.pkl')

    # Save the scaler to a pickle file
    joblib.dump(scaler, 'scaler.pkl')

    # Save the metrics in the database (Django model)
    LR.objects.create(
        name="Logistic Regression Model",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1
    )

    # Fetch the latest result to pass to the template
    results = LR.objects.last()

    # Prepare the results for rendering
    context = {
        'name': results.name,
        'accuracy': round(results.accuracy, 2),
        'precision': round(results.precision, 2),
        'recall': round(results.recall, 2),
        'f1_score': round(results.f1_score, 2)
    }

    # Render the results to the template
    return render(request, 'admin/LR_alg.html', context)



