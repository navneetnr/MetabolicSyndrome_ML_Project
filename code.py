import pandas as pd
from sklearn.preprocessing import LabelEncoder

#load dataset   
df=pd.read_csv("C:\\Users\\LENOVO\\Desktop\\Sem-5\\ml ca2\\Metabolic_Syndrome.csv")
print(df)
df.info()
print(df.describe())
print(df.shape)
print(df.isnull().sum())


# Apply nunique 
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


x=df.drop("MetabolicSyndrome",axis=1)
y=df["MetabolicSyndrome"]

print(df["MetabolicSyndrome"].value_counts())


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


#balance dataset
sm=SMOTE(random_state=42)
x_train,y_train=sm.fit_resample(x,y)
print(y_train.value_counts())

#Standered Scaler
num_cols = ["Age", "Income", "WaistCirc", "BMI", "HDL", "Triglycerides"] 
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train[num_cols])
x_test_scaled = scaler.transform(x_test[num_cols])

x_train[num_cols] = x_train_scaled
x_test[num_cols] = x_test_scaled



#using logistic regression
lr=LogisticRegression(max_iter=5000, solver='saga')
lr.fit(x_train,y_train)
acc=lr.score(x_test,y_test)*100
print(f"Test Accuracy: {acc:.2f}%")

sns.heatmap(x_train.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()





