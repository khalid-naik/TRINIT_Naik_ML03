import numpy as np;
import pandas as pd;
# import matplotlib.pyplot as plt;
from sklearn.model_selection import train_test_split;
from sklearn import metrics;
from sklearn import neural_network;
import joblib;

df = pd.read_csv(r"C:/Users/khalid naik/store2/NIT_Tirchi/cropprediction/Crop_recommendation.csv");
# print(df);

x = df[['N','P','K','temperature','humidity','ph','rainfall']];
y = df[['label']];
# print(x,y);

from sklearn.preprocessing import StandardScaler;
scaler = StandardScaler();
x = scaler.fit_transform(x);
# print(x);

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42);

#model
clf = neural_network.MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, alpha=0.0001,solver='adam', random_state=42, tol=0.0001);
clf.fit(x_train,y_train);

#evaluating the model
train_score = clf.score(x_train,y_train);
test_score = clf.score(x_test,y_test);
print("train_score = ",train_score);
print("test_score = ",test_score);

new_data = [[104,18,30,23.603016,60.396475,6.779833,140.937041]];
new_data = scaler.transform(new_data);
predictions = clf.predict(new_data);
print('Predictions:', predictions);

joblib.dump(clf, "neural_network.joblib");
joblib.dump(scaler, 'scaler.joblib');