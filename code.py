import pandas as pd

#load dataset
df=pd.read_csv("C:\\Users\\LENOVO\\Desktop\\Sem-5\\ml ca2\\Metabolic_Syndrome.csv")
print(df)
print(df.info())
print(df.describe())
print(df.shape)
print(df.isnull().sum())

