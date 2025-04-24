import pandas as pd


# Change according to your file path
filePath = 'D:\\Projet\\SpamWise\\DataBase\\Untreated_Data\\SmsScam1_DataBase.txt'


df = pd.read_csv(filePath, sep="\t", header=None, names=["SpamLabel", "Content"])


df['IsSpam'] = df['SpamLabel'].replace({'ham': 0, 'spam': 1})
df['Type'] = 'sms'
df['LengthContent'] = df['Content'].apply(len)
df['Has-Link'] = df['Content'].str.contains(r'h[t]{1,2}p[s]?://|www\.', case=False, regex=True).astype(int)
df = df[['Type', 'Content', 'LengthContent', 'Has-Link', 'IsSpam']]


print(df.head())
#df.to_csv("D:\Projet\SpamWise\DataBase\Treated_Data\\SmsScam1_DataBaseGood.csv", index=False)
