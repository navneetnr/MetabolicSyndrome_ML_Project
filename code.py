import pandas as pd

#load dataset
df=pd.read_csv("C:\\Users\\LENOVO\\Desktop\\Sem-5\\ml ca2\\Metabolic_Syndrome.csv")
print(df)
df.info()
print(df.describe())
print(df.shape)
print(df.isnull().sum())
#data cleaning
df["Marital"].fillna(df["Marital"].mode()[0], inplace=True)
print(df.isnull().sum())
df["Income"].fillna(df["Income"].median(),inplace=True)
print(df.isnull().sum())
df["WaistCirc"].fillna(df["WaistCirc"].median(),inplace=True)
print(df.isnull().sum())
df["BMI"].fillna(df["BMI"].median(),inplace=True)
print(df.isnull().sum())
