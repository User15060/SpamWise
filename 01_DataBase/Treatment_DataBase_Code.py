import pandas as pd
from pathlib import Path
import glob
from enum import Enum
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

BASE_DIR = Path(__file__).parent
UNTREATED_DIR = glob.glob(f'{BASE_DIR}\\02_Untreated_DataBase\\*.csv')
TREATED_DIR = glob.glob(f'{BASE_DIR}\\03_Treated_DataBase\\')

class FileType(Enum): 
    URL = 'url'
    SMS = 'sms'
    EMAIL = 'email'

def TreatmentUrl(df):
    df['IsSpam'] = df['label']
    df['Content'] = df['URL']
    df['Type'] = FileType.URL.value 

def TreatmentSms(df):
    df['IsSpam'] = df['CLASS']
    df['Content'] = df['CONTENT']
    df['Type'] = FileType.SMS.value  

def TreatmentEmail(df):
    if 'label_num' in df.columns:
        df['IsSpam'] = df['label_num'].replace({'ham': 0, 'spam': 1})
    elif 'is_spam' in df.columns:
        df['IsSpam'] = df['is_spam'].replace({'ham': 0, 'spam': 1})

    if 'text' in df.columns:
        df['Content'] = df['text']
    elif 'message_content' in df.columns:
        df['Content'] = df['message_content']
    df['Type'] = FileType.EMAIL.value  

def TreatmentMain(df):
    df['LengthContent'] = df['Content'].apply(len)
    df['Has-Link'] = df['Content'].str.contains(r'h[t]{1,2}p[s]?://|www\.', case=False, regex=True).astype(int)
    df = df[['Type', 'Content', 'LengthContent', 'Has-Link', 'IsSpam']]
    return df

csvFiles_Dir_Untreated = glob.glob(f'{UNTREATED_DIR}\\*.csv')

for filePath in csvFiles_Dir_Untreated:
    fileName = Path(filePath).name

    df = pd.read_csv(filePath)

    if FileType.SMS.value in fileName.lower():  
        TreatmentSms(df)
    elif FileType.EMAIL.value in fileName.lower(): 
        TreatmentEmail(df)
    elif FileType.URL.value in fileName.lower(): 
        TreatmentUrl(df)

    df = TreatmentMain(df)

    df.to_csv(f'{TREATED_DIR}\\{fileName}', index=False)


csv_Dir_Treated = glob.glob(f'{TREATED_DIR}\\*.csv')

if csv_Dir_Treated:
    df_combine = pd.concat([pd.read_csv(f) for f in csv_Dir_Treated], ignore_index=True)

    #print(df_combine.columns)

    df_combine.to_csv('{TREATED_DIR}\\FinalTreated_Database\\Treated_Database.csv', index=False)
else:
    print("Aucun fichier traité trouvé dans le répertoire '03_Treated_DataBase'.")





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
