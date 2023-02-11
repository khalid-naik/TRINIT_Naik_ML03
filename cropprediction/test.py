from joblib import load;
from sklearn.preprocessing import StandardScaler;


model_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/neuralnetwork.joblib';
scaler_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/sccaler.joblib';
model = load(model_path);
scaler = load(scaler_path);


# new_data = [[90,42,43,20.879774,82.00274,6.502985,202.9355], [71,54,16,22.6136,63.69071,5.749914,87.75954], [40,72,77,17.02498,16.98861,7.485996,88.55123]]
new_data = [[104 , 18 , 30  ,  23.603016 , 60.396475 , 6.779833 , 140.937041]];
new_data = scaler.transform(new_data);
predicted_class = model.predict(new_data)

print(predicted_class)