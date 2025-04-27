import pandas as pd
from pathlib import Path
import glob
from enum import Enum
import nltk
import re
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download if you don't have it 
#nltk.download(['wordnet', 'stopwords'])

class FileType(Enum):
    URL = 'url'
    SMS = 'sms'
    EMAIL = 'email'

class DataProcessor:
    def __init__(self):
        self.BaseDir = Path(__file__).parent
        self.UntreatedDir = self.BaseDir / "02_Untreated_DataBase"
        self.TreatedDir = self.BaseDir / "03_Treated_DataBase"
        self.TreatedExternalDir = self.TreatedDir / "External_ListSources"
        self.SpamWordList = self.LoadSpamWords()
        self.Lemmatizer = WordNetLemmatizer()
        self.StopWords = set(stopwords.words('english'))
        self.le = LabelEncoder()


    def LoadSpamWords(self):
        spamListFile = self.TreatedExternalDir / "SpamListWords_DataBase.txt"
        if spamListFile.exists():
            with open(spamListFile, "r", encoding="latin-1") as file:
                return {line.strip() for line in file}
        return set()

    def ProcessFiles(self):
        csvFiles = glob.glob(str(self.UntreatedDir / '*.csv'))
        if not csvFiles:
            print("No file found.")
            return None

        processedDfs = []
        for filePath in csvFiles:
            df = pd.read_csv(filePath)
            fileType = self.DetectFileType(Path(filePath).name)
            
            if fileType == FileType.URL:
                df = self.ProcessUrl(df)
            elif fileType == FileType.SMS:
                df = self.ProcessSms(df)
            elif fileType == FileType.EMAIL:
                df = self.ProcessEmail(df)
            
            processedDfs.append(self.ProcessMain(df))

        return pd.concat(processedDfs, ignore_index=True) if processedDfs else None

    def DetectFileType(self, fileName):
        fileName = fileName.lower()
        if 'url' in fileName:
            return FileType.URL
        elif 'sms' in fileName:
            return FileType.SMS
        elif 'email' in fileName:
            return FileType.EMAIL
        return None

    def ProcessUrl(self, df):
        df['IsSpam'] = df['label']
        df['Content'] = df['URL']
        df['Type'] = FileType.URL.value
        return df

    def ProcessSms(self, df):
        df['IsSpam'] = df['CLASS']
        df['Content'] = df['CONTENT']
        df['Type'] = FileType.SMS.value
        return df

    def ProcessEmail(self, df):
        if 'label_num' in df.columns:
            df['IsSpam'] = df['label_num'].replace({'ham': 0, 'spam': 1})
        elif 'is_spam' in df.columns:
            df['IsSpam'] = df['is_spam'].replace({'ham': 0, 'spam': 1})

        df['Content'] = df['text'] if 'text' in df.columns else df['message_content']
        df['Type'] = FileType.EMAIL.value
        return df

    def ProcessMain(self, df):
        df['LengthContent'] = df['Content'].str.len()
        df['HasLink'] = df['Content'].str.contains(r'http\S+|www\.', regex=True).astype(int)
        return df[['Type', 'Content', 'LengthContent', 'HasLink', 'IsSpam']]

    def CleanContent(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", '__URL__', text)
        text = re.sub(r'\S+@\S+', '__EMAIL__', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-z\s_]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def LemmatizeText(self, text):
        words = [self.Lemmatizer.lemmatize(word) 
                for word in text.split() 
                if word not in self.StopWords]
        return ' '.join(words)

    def CountSpamWords(self, text):
        return sum(1 for word in set(text.split()) if word in self.SpamWordList)

    def AnalyzeTextFeatures(self, text):
        text = str(text)
        return {
            'NumberPunctuations': len(re.findall(r'[^\w\s]', text)),
            'NumberUppercase': len(re.findall(r'[A-Z]', text)),
            'NumberLowercase': len(re.findall(r'[a-z]', text)),
            'NumberDigits': len(re.findall(r'\d', text))
        }

    def CreateFinalDataset(self, df):
        df['CleanContent'] = df['Content'].apply(self.CleanContent)
        df['CleanContent'] = df['CleanContent'].apply(self.LemmatizeText)
        
        features = df['Content'].apply(lambda x: pd.Series(self.AnalyzeTextFeatures(x)))
        df = pd.concat([df, features], axis=1)
        
        df['LengthContentModified'] = df['CleanContent'].str.len()
        df['SpamWordCount'] = df['CleanContent'].apply(self.CountSpamWords)
        
        df['TypeEncoded'] = self.le.fit_transform(df['Type'])

        return df[['Type', 'TypeEncoded', 'Content', 'LengthContent', 'HasLink', 'CleanContent',
                'NumberPunctuations', 'NumberUppercase', 'NumberLowercase', 'NumberDigits',
                'LengthContentModified', 'SpamWordCount', 'IsSpam']]
        

    def SaveFinalDataset(self, df):
        output_dir = self.TreatedDir / "FinalTreated_Database"
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "Treated_Database_V1.csv", index=False)


if __name__ == "__main__":
    processor = DataProcessor()
    combined_df = processor.ProcessFiles()
    
    if combined_df is not None:
        final_df = processor.CreateFinalDataset(combined_df)
        processor.SaveFinalDataset(final_df)
        