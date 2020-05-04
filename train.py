import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('echocardiogram.data', sep=',', header=None)

for i in list(df):
    df = df[df[i] != '?']

df = df.drop([10], axis=1)

print(df.shape)
df = df.drop_duplicates()
print(df.shape)


plt.hist(df[list(df)[-1]])

X = df[list(df)[:-1]]
y = df[list(df)[-1]]


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

std = StandardScaler()
model1 = SVC()
model1.fit(std.fit_transform(X_train), y_train)

print(accuracy_score(model1.predict(X_test), y_test))

model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)

print(accuracy_score(model2.predict(X_test), y_test))

model3 = GradientBoostingClassifier()
model3.fit(X_train, y_train)

print(accuracy_score(model3.predict(std.transform(X_test)), y_test))

model4 = RandomForestClassifier()
model4.fit(X_train, y_train)

print(accuracy_score(model4.predict(std.transform(X_test)), y_test))
