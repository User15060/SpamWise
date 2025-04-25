import pandas as pd
import glob
import os


# Change according to your folder path
folderPath = 'D:\\Projet\\SpamWise\\SpamWise\\data_base\\treated_database\\treated_database'
fichiers_csv = glob.glob(os.path.join(folderPath, '*.csv'))

df_combine = pd.concat([pd.read_csv(f) for f in fichiers_csv], ignore_index=True)
print(df_combine.columns)

#df_combine.to_csv('D:\\Projet\\SpamWise\\SpamWise\data_base\\treated_database\\main_database\\DataBaseGood.csv', index=False)
