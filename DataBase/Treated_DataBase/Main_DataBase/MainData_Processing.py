import pandas as pd
import glob
import os


# Change according to your folder path
folderPath = 'D:\\Projet\\SpamWise\\DataBase\\Treated_Data'
fichiers_csv = glob.glob(os.path.join(folderPath, '*.csv'))

df_combine = pd.concat([pd.read_csv(f) for f in fichiers_csv], ignore_index=True)
print(df_combine.columns)

#df_combine.to_csv('D:\\Projet\\SpamWise\\DataBase\\Treated_Data\\DataBaseGood.csv', index=False)
