import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# load Dataset
df = pd.read_csv("C:\\Users\\LENOVO\\Desktop\\Sem-5\\ml ca2\\Metabolic_Syndrome.csv")

print(df.info())
print(df.describe())
print(df.shape)
print(df.isnull().sum())
print(df.nunique())

#data cleaning
df["Marital"]=df["Marital"].fillna(df["Marital"].mode()[0])
df["Income"]=df["Income"].fillna(df["Income"].median())
df["WaistCirc"]=df["WaistCirc"].fillna(df["WaistCirc"].median())
df["BMI"]=df["BMI"].fillna(df["BMI"].median())

#using dummies for transform value
df = pd.get_dummies(df, columns=["Sex", "Race"], drop_first=True, dtype=int)
print(df)

#using label encoder
le = LabelEncoder()
df["Marital"] = le.fit_transform(df["Marital"])

x = df.drop("MetabolicSyndrome", axis=1)
y = df["MetabolicSyndrome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# balance dataset
sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)

# Standard Scaling
num_cols = ["Age", "Income", "WaistCirc", "BMI", "HDL", "Triglycerides"]
scaler = StandardScaler()

x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])

#using co-realtion 
plt.figure(figsize=(12,8))
sns.heatmap(x_train.corr(), cmap="coolwarm", annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Logistic Regression
lr = LogisticRegression(max_iter=1000, solver="saga")
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

print("\nLogistic Regression")
print("Accuracy:", lr.score(x_test, y_test)*100)
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()


# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print("\nRandom Forest")
print("Accuracy:", rf.score(x_test, y_test)*100)
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest - Confusion Matrix")
plt.show()

# SVM
svm = SVC(kernel="rbf", C=1.0)
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)

print("\nSVM")
print("Accuracy:", svm.score(x_test, y_test)*100)
print("Recall:", recall_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm))

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt="d", cmap="Oranges")
plt.title("SVM - Confusion Matrix")
plt.show()

# XGBoost
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)

print("\nXGBoost")
print("Accuracy:", xgb.score(x_test, y_test)*100)
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))

plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt="d", cmap="Purples")
plt.title("XGBoost Confusion Matrix")
plt.show()

#graph between unbalace and balance graph
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Original DataSet")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x=y_train)
plt.title(" Balanced DataSet")
plt.show()

#use histogram to find frequency of data
x_train[num_cols].hist(figsize=(12,8), bins=20)
plt.suptitle("Numerical Feature Distributions")
plt.show()
