from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from pathlib import Path
import joblib
import pandas as pd
import sys


class SpamClassfier:
    def __init__(self):
            BaseCSVDir = Path(__file__).parent.parent.parent
            self.BaseModelDir = Path(__file__).parent
            self.final_df = BaseCSVDir / "01_DataBase\\03_Treated_DataBase\\FinalTreated_Database\\Treated_Database_V2.csv"
            self.model = RandomForestClassifier(n_estimators=1000, random_state=42)
            self.tfidf_content = TfidfVectorizer(max_features=200)
            self.tfidf_cleancontent = TfidfVectorizer(max_features=200)


    def Train(self, df):  
        try:      
            tfidf_content_matrix = self.tfidf_content.fit_transform(df['Content'].fillna(''))
            tfidf_cleancontent_matrix = self.tfidf_cleancontent.fit_transform(df['CleanContent'].fillna(''))
            
            numeric_features = df[['LengthContent', 'HasLink', 'NumberPunctuations', 
                                'NumberUppercase', 'NumberLowercase', 'NumberDigits', 
                                'LengthContentModified', 'SpamWordCount', 'TypeEncoded']]
            
            X = hstack([numeric_features, tfidf_content_matrix, tfidf_cleancontent_matrix])
            y = df['IsSpam']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            print(classification_report(y_test, y_pred))
            print(confusion_matrix(y_test, y_pred))
        except KeyError as e:
            print(f"Error: Missing column in the DataFrame -> {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error during training: {e}")
            sys.exit(1)

    def SaveModel(self, filename='spamClassifierV2.pkl'):
        try:
            full_path = self.BaseModelDir / filename
            joblib.dump({
                'model': self.model,
                'tfidf_content': self.tfidf_content,
                'tfidf_cleancontent': self.tfidf_cleancontent
            }, full_path)
            print(f"Model successfully saved to: {full_path}")
        except Exception as e:
            print(f"Error while saving the model: {e}")
            sys.exit(1)


if __name__ == "__main__":
    classifier = SpamClassfier()

    if not classifier.final_df.exists():
         print(f"Error: File '{classifier.file_df}'not found")
         sys.exit(1)

    try:
        df = pd.read_csv(classifier.final_df)
    except Exception as e:
         print(f"Error while reading the CSV file: {e}")

    classifier.Train(df)
    classifier.SaveModel()