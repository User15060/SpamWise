import pandas as pd


# Change according to your file path
filePath = 'D:\\Projet\\SpamWise\\DataBase\\Untreated_Data\\EmailScam1_DataBase.csv'


df = pd.read_csv(filePath)
df = df[["is_spam", "message_content"]]

df['IsSpam'] = df['is_spam'].replace({'ham': 0, 'spam': 1})
df['Type'] = 'email'
df['Content'] = df['message_content']
df['LengthContent'] = df['Content'].apply(len)
df['Has-Link'] = df['Content'].str.contains(r'h[t]{1,2}p[s]?://|www\.', case=False, regex=True).astype(int)
df = df[['Type', 'Content', 'LengthContent', 'Has-Link', 'IsSpam']]


print(df.head())
#df.to_csv("D:\Projet\SpamWise\DataBase\Treated_Data\\EmailScam1_DataBaseGood.csv", index=False)
