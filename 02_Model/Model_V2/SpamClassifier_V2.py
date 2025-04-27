from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import joblib

class SpamClassfier:
    def __init__(self):
            self.BaseDir = Path(__file__).parent
            #FAIRE CHEMIN VERS GoodData
            # GESTION ERREUR + VARIABLE
            # REGARDE DOWNLOAD
            self.FinalDf = output_dir / "Treated_Database_V1.csv"
            self.model = RandomForestClassifier(n_estimators=1000, random_state=42)
            self.tfidf_content = TfidfVectorizer(max_features=200)
            self.tfidf_cleancontent = TfidfVectorizer(max_features=200)


    def train(self, df):        
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

    def save_model(self, filename='spamClassifierV2.pkl'):
        joblib.dump({
            'model': self.model,
            'tfidf_content': self.tfidf_content,
            'tfidf_cleancontent': self.tfidf_cleancontent
        }, filename)


if __name__ == "__main__":    
    if combined_df is not None:
        final_df = processor.create_final_dataset(combined_df)
        processor.save_final_dataset(final_df)
        
        classifier = SpamClassifier()
        classifier.train(final_df)
        classifier.save_model()