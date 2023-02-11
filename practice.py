import csv
from joblib import load

model_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/neural_network.joblib';
model = load(model_path);
scaler_path = 'C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/scaler.joblib';
scaler = load(scaler_path);

# Open the CSV file and read in the data
with open('C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/Crop_recommendation _update2.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

# Loop through the rows and check for the matching first value
my_list = []
for row in data:
    if row[0] == 'NICOBAR':
        # Store the rest of the values in that row
        other_values = row[1:8]
        my_list.append(other_values)
my_list = my_list[0];
# print(my_list)
print(len(my_list));
N = float(my_list[0]);
P = float(my_list[1]);
K = float(my_list[2]);
temperature = float(my_list[3]);
humidity = float(my_list[4]);
ph = float(my_list[5]);
rainfall = float(my_list[6]);

data = [];
data.append(N);
data.append(P);
data.append(K);
data.append(temperature);
data.append(humidity);
data.append(ph);
data.append(rainfall);

# print(data);
inputs = [data[0], data[1], data[2],data[3], data[4], data[5],data[6]];
print(inputs);
input2 = scaler.transform([inputs])[0];
print(input2);
# prediction = model.predict([data])[0];
# print(prediction);
