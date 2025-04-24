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
df = pd.read_csv("D:\\Projet\\SpamWise\\DataBase\\Treated_DataBase\\Main_DataBase\\DataBaseGood.csv")


filePath = "D:\\Projet\\SpamWise\\DataBase\\Others\\SpamListWords\\SpamListWords.txt"
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

# Lemmatisation du texte
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Ignorer les stopwords
    return " ".join(lemmatized_words)

# Compter les mots de la liste de spam dans un texte
def count_spam_words(text, spam_words):
    words = set(text.split())  # Utilisation d'un set pour éviter les doublons
    return sum(1 for word in words if word in spam_words)

# Appliquer les fonctions sur ton DataFrame
df['CleanContent'] = df['Content'].apply(clean_text_full)  # Nettoyage de base
df['CleanContent'] = df['CleanContent'].apply(lemmatize_text)  # Lemmatisation et stopwords
df['LengthContent'] = df['Content'].apply(lambda x: len(str(x)))  # Longueur du contenu
df['Has-Link'] = df['Content'].str.contains(r"http\S+|www\S+", regex=True).astype(int)  # Présence de liens
df['LengthContentModified'] = df['CleanContent'].apply(lambda x: len(x))  # Longueur après nettoyage
df['ListWords'] = df['CleanContent'].apply(lambda x: count_spam_words(x, spamWordList))  # Nombre de mots de spam dans le contenu

df.to_csv('D:\\Projet\\SpamWise\\DataBase\\Treated_DataBase\\Main_DataBase\\DataBaseGood1.csv', index=False)

# Séparation des features et de la cible
# X = df[['CleanContent', 'LengthContent', 'Has-Link', 'LengthContentModified', 'ListWords']]
# y = df['IsSpam']