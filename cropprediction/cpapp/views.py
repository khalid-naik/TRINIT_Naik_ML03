
from django.shortcuts import render, redirect
from django.http import JsonResponse
from joblib import load
import csv
from django.http import JsonResponse

def input_form(request):
    return render(request, 'input.html')

def predict(request):
    if request.method == 'POST':
        model_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/neural_network.joblib'
        model = load(model_path)
        scaler_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/scaler.joblib'
        scaler = load(scaler_path)

        district_name = request.POST.get('district', '')
        month_name = request.POST.get('month', '')

        district_data = None

        with open('C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/Crop_recommendation _update3.csv', 'r') as file:
            reader = csv.reader(file)
            data = list(reader)

        for row in data:
            if row[0] == district_name and row[1] == month_name: 
                district_data = row[2:9]
                break

        if district_data is None:
            return JsonResponse({'error': 'Invalid district or month name'})

        inputs = [float(x) for x in district_data]
        inputs = scaler.transform([inputs])[0]
        prediction = model.predict([inputs])[0]

        return JsonResponse({'prediction': prediction})
    else:
        return redirect('input')

