import pandas as pd


# Change according to your file path
filePath = 'D:\\Projet\\SpamWise\\DataBase\\Untreated_Data\\SmsScam2_DataBase.csv'


df = pd.read_csv(filePath)
df = df[["CLASS", "CONTENT"]]

df['IsSpam'] = df['CLASS']
df['Type'] = 'sms'
df['Content'] = df['CONTENT']
df['LengthContent'] = df['Content'].apply(len)
df['Has-Link'] = df['Content'].str.contains(r'h[t]{1,2}p[s]?://|www\.', case=False, regex=True).astype(int)
df = df[['Type', 'Content', 'LengthContent', 'Has-Link', 'IsSpam']]


print(df.head())
#df.to_csv("D:\Projet\SpamWise\DataBase\Treated_Data\\SmsScam2_DataBaseGood.csv", index=False)
