import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
#from google.colab import files

df = pd.read_csv("/content/Treated_Database_V1.csv")

X = df[['LengthContent', 'HasLink', 'LengthContentModified', 'SpamWordCount']]
y = df['IsSpam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

#joblib.dump(model, 'spamClassifierV1.pkl')  
#files.download('spamClassifierV1.pkl')
