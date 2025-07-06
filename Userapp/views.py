from django.shortcuts import render,redirect
from Mainapp.models import *
from Userapp.models import Dataset
from Adminapp.models import *
from django.contrib import messages
import time
from django.core.paginator import Paginator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.core.files.storage import default_storage
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.shortcuts import render, redirect
from .models import User  # Ensure your models are imported
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def userdashboard(req):
    images_count =  User.objects.all().count()
    print(images_count)
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id)
    return render(req,'user/user-dashboard.html')
  



def userlogout(req):
    user_id = req.session["User_id"]
    user = User.objects.get(User_id = user_id) 
    t = time.localtime()
    user.Last_Login_Time = t
    current_time = time.strftime('%H:%M:%S', t)
    user.Last_Login_Time = current_time
    current_date = time.strftime('%Y-%m-%d')
    user.Last_Login_Date = current_date
    user.save()
    messages.info(req, 'You are logged out..')
    return redirect('index')




import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from django.shortcuts import render

# This function will load the trained model and scaler, then make a prediction
def predict_ransomware(request):
    if request.method == 'POST':
        # Get the data from the form (replace 'form_field' with actual input names)
        # Assuming you have a form in HTML to input these values
        file_data = {
            'Machine': float(request.POST['Machine']),
            'DebugSize': float(request.POST['DebugSize']),
            'DebugRVA': float(request.POST['DebugRVA']),
            'MajorImageVersion': float(request.POST['MajorImageVersion']),
            'MajorOSVersion': float(request.POST['MajorOSVersion']),
            'ExportRVA': float(request.POST['ExportRVA']),
            'ExportSize': float(request.POST['ExportSize']),
            'IatVRA': float(request.POST['IatVRA']),
            'MajorLinkerVersion': float(request.POST['MajorLinkerVersion']),
            'MinorLinkerVersion': float(request.POST['MinorLinkerVersion']),
            'NumberOfSections': float(request.POST['NumberOfSections']),
            'SizeOfStackReserve': float(request.POST['SizeOfStackReserve']),
            'DllCharacteristics': float(request.POST['DllCharacteristics']),
            'ResourceSize': float(request.POST['ResourceSize']),
            'BitcoinAddresses': float(request.POST['BitcoinAddresses'])
        }

        # Convert the input data into a pandas DataFrame for prediction
        input_data = pd.DataFrame([file_data])

        # Load the trained model and scaler
        model = joblib.load('svm_model.pkl')  # Load your saved model
        scaler = joblib.load('scaler.pkl')  # Load the saved scaler

        # Scale the input data (same as the training data)
        input_data_scaled = scaler.transform(input_data)

        # Make prediction using the trained model
        prediction = model.predict(input_data_scaled)

        # Check the prediction
        if prediction == 0:
            result = "Benign"
        else:
            result = "Ransomware"

        # Display the result on the webpage
        context = {'prediction': result}

        return render(request, 'user/predict_ransomware.html', context)

    return render(request, 'user/predict_ransomware.html')





















        





    
   

