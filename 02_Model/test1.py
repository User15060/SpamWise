import joblib
import pandas as pd

model = joblib.load('D:\\Projet\\SpamWise\\SpamWise\\model\\spam_classifier.pkl')


new_data = pd.read_csv('new_data.csv')

X_new = new_data[['LengthContent', 'Has-Link', 'LengthContentModified', 'ListWords']]

predictions = model.predict(X_new)

new_data['Prediction'] = ['SPAM' if pred == 1 else 'NON-SPAM' for pred in predictions]

print(new_data[['Content', 'Prediction']])
