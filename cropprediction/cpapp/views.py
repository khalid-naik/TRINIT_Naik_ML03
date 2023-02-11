from django.shortcuts import render, redirect
from django.http import JsonResponse
from joblib import load

def input_form(request):
    return render(request, 'input.html')

def predict(request):
    if request.method == 'POST':
        # Load the trained model into memory
        model_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/neural_network.joblib';
        model = load(model_path);
        scaler_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/scaler.joblib';
        scaler = load(scaler_path);

        # Parse the input data
        data = request.POST.dict()
        inputs = [float(data['N']), float(data['P']), float(data['K']),float(data['temperature']), float(data['humidity']), float(data['ph']),float(data['rainfall'])];

        input2 = [float(x) for x in inputs]
        input2 = scaler.transform([input2])[0]

        # inputs = scaler.transform(inputs);
        # Use the model to make a prediction
        prediction = model.predict([input2])[0]

        # Return the prediction to the client
        return JsonResponse({'prediction': prediction})
    else:
        # If the HTTP method is not POST, redirect to the input form
        return redirect('input')
