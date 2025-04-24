import os


# Chemin du fichier
filePath = "D:\\Projet\\SpamWise\\DataBase\\Others\\SpamListWords\\SpamListWords.txt"
spamWords = set()


with open(filePath, "r") as fichier:
    for ligne in fichier:
        mot = ligne.strip().lower()
        spamWords.add(mot)

with open(filePath, "w") as fichier:
    for mot in spamWords:
        fichier.write(mot + "\n") 
