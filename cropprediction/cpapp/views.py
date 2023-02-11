
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

        # district_name, month_name = request.POST.get('district', 'month')
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


# def predict(request):
    # if request.method == 'POST':
    #     # Load model and scaler
    #     model_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/neural_network.joblib';
    #     model = load(model_path);
    #     scaler_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/scaler.joblib';
    #     scaler = load(scaler_path);

    #     # Get district name from form input
    #     district_name,month_name = request.POST.get('district','month')

    #     # Load data from CSV file
    #     with open('C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/Crop_recommendation _update3.csv', 'r') as file:
    #         reader = csv.reader(file)
    #         data = list(reader)

    #     # Loop through the rows and check for the matching district name
    #     district_data = None
    #     for row in data:
    #         if row[0] == district_name & row[1] == month_name : 
    #             district_data = row[2:9]
    #             break

    #     # If district data is found, preprocess input and make prediction
    #     if district_data:
    #         inputs = [float(x) for x in district_data]
    #         inputs = scaler.transform([inputs])[0]
    #         prediction = model.predict([inputs])[0]

    #         # Return predicted output as JSON response
    #         return JsonResponse({'prediction': prediction})

    # # If district data is not found or request method is not POST, return error message
    # return JsonResponse({'error': 'Invalid district name or request method.'})




    # if request.method == 'POST':
    #     model_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/neural_network.joblib';
    #     model = load(model_path);
    #     scaler_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/scaler.joblib';
    #     scaler = load(scaler_path);

    #     district = request.POST.get('district')

    #     with open('C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/Crop_recommendation _update2.csv', 'r') as file:
    #         reader = csv.reader(file)
    #         data = list(reader)

    #     # Loop through the rows and check for the matching district value
    #     my_list = []
    #     for row in data:
    #         if row[0] == district:
    #             # Store the rest of the values in that row
    #             other_values = row[1:8]
    #             my_list.append(other_values)

    #     # If the district was not found, return an error message
    #     if not my_list:
    #         return JsonResponse({'error': 'District not found'})

    #     # Get the first row of the list
    #     my_list = my_list[0]

    #     # Convert the values to float
    #     N = float(my_list[0])
    #     P = float(my_list[1])
    #     K = float(my_list[2])
    #     temperature = float(my_list[3])
    #     humidity = float(my_list[4])
    #     ph = float(my_list[5])
    #     rainfall = float(my_list[6])

    #     # Scale the input values
    #     inputs = [N, P, K, temperature, humidity, ph, rainfall]
    #     inputs_scaled = scaler.transform([inputs])[0]

    #     # Make the prediction
    #     prediction = model.predict([inputs_scaled])[0]

    #     # Return the predicted crop
    #     return JsonResponse({'prediction': prediction})

    # else:
    #     return redirect('input')


