import pandas as pd


# Change according to your file path
filePath = 'D:\\Projet\\SpamWise\\DataBase\\Untreated_Data\\UrlScam1_DataBase.csv'


df = pd.read_csv(filePath)

df['IsSpam'] = df['label']
df['Type'] = 'url'
df['Content'] = df['URL']
df['LengthContent'] = df['Content'].apply(len)
df['Has-Link'] = df['Content'].str.contains(r'h[t]{1,2}p[s]?://|www\.', case=False, regex=True).astype(int)
df = df[['Type', 'Content', 'LengthContent', 'Has-Link', 'IsSpam']]


print(df.head())
#df.to_csv("D:\Projet\SpamWise\DataBase\Treated_Data\\UrlScam1_DataBaseGood.csv.csv", index=False)
