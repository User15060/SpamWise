import pandas as pd
import glob
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('wordnet')
nltk.download('stopwords')
spamWordList = set()
df = pd.read_excel("D:\\Projet\\SpamWise\\SpamWise\\model\\testdata.ods", engine="odf")


filePath = "D:\\Projet\\SpamWise\\SpamWise\\data_base\\untreated_database\\spam_list_words\\SpamListWords_DataBase.txt"
with open (filePath, "r") as folder:
    for ligne in folder:
        spamWordList.add(ligne.strip())

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text_full(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", '__URL__', text)   # URL -> _url_
    text = re.sub(r'\S+@\S+', '__EMAIL__', text)        # MAIL -> _mail_
    text = re.sub(r'\d+', '', text)                     # URL -> _url_
    text = re.sub(r'[^a-z\s_]', '', text)               # URL -> _url_ 
    text = re.sub(r'\s+', ' ', text).strip()            # URL -> _url_
    return text 

def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Ignorer les stopwords
    return " ".join(lemmatized_words)

def count_spam_words(text, spam_words):
    words = set(text.split())  # Utilisation d'un set pour éviter les doublons
    return sum(1 for word in words if word in spam_words)

df['CleanContent'] = df.apply(lambda row: lemmatize_text(clean_text_full(row['Content'])) if row['Type'].lower() != "url" else "", axis=1)
df['LengthContent'] = df['Content'].apply(lambda x: len(str(x)))  
df['Has-Link'] = df['Content'].str.contains(r"http\S+|www\S+", regex=True).astype(int)  
df['LengthContentModified'] = df['CleanContent'].apply(lambda x: len(x) if x else 0)
df['ListWords'] = df['CleanContent'].apply(lambda x: count_spam_words(x, spamWordList))  
print(df.head())
df.to_csv('D:\\Projet\\SpamWise\\SpamWise\\data_base\\treated_database\\main_database\\DataBase1.csv', index=False)

# Séparation des features et de la cible
# X = df[['CleanContent', 'LengthContent', 'Has-Link', 'LengthContentModified', 'ListWords']]
# y = df['IsSpam']