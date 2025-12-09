import pandas as pd
from sklearn.preprocessing import LabelEncoder

#load dataset   
df=pd.read_csv("C:\\Users\\LENOVO\\Desktop\\Sem-5\\ml ca2\\Metabolic_Syndrome.csv")
print(df)
df.info()
print(df.describe())
print(df.shape)
print(df.isnull().sum())


# Apply nunique to know unique value 
print("unique value")
print(df.nunique())

#data cleaning    
df["Marital"]=df["Marital"].fillna(df["Marital"].mode()[0])

df["Income"]=df["Income"].fillna(df["Income"].median())

df["WaistCirc"]=df["WaistCirc"].fillna(df["WaistCirc"].median())

df["BMI"]=df["BMI"].fillna(df["BMI"].median())
print(df.isnull().sum())

#using dummies for transform value       
df=pd.get_dummies(df,columns=["Sex"],drop_first=True,dtype=int)
print(df)


#Using labelEncoder
le=LabelEncoder()
df["Marital"]=le.fit_transform(df["Marital"])
print(df)

