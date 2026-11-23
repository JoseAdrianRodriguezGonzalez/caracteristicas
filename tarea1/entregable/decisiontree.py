#Quinto archivo a revisar: Decision tree
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.utils import shuffle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('features_dataset_completo.csv')
#df=pd.read_csv('dataset_ordenado.csv')
X=df.drop(columns='category')
Y=df['category']
#X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42,stratify=Y)
data=shuffle(df,random_state=42)
test_indices = []
train_indices = []
n_test_por_clase = min(21, Y.value_counts().min())
for clase in df['category'].unique():
    indices_clase = df[df['category'] == clase].index
    test_indices_clase = indices_clase[:n_test_por_clase] 
    train_indices_clase = indices_clase[n_test_por_clase:]
    test_indices.extend(test_indices_clase)
    train_indices.extend(train_indices_clase)
X_train = df.loc[train_indices].drop(columns=['category'])
y_train = df.loc[train_indices]['category']
X_test = df.loc[test_indices].drop(columns=['category'])
y_test = df.loc[test_indices]['category']

modelo=DecisionTreeClassifier(random_state=42,max_depth=5,min_samples_split=5,ccp_alpha=0.005,criterion='gini')
modelo.fit(X_train,y_train)
y_pred_train=modelo.predict(X_train)
accuracy_train=accuracy_score(y_train,y_pred_train)
y_pred=modelo.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy train:{accuracy_train}")
print(f"Accuracy test:{accuracy}")
print(f"Diferencia: {accuracy_train-accuracy}")
print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=modelo.classes_, yticklabels=modelo.classes_)
plt.xlabel('Predicho')
plt.ylabel('Actual')
plt.title('Matriz de confusión')
#plt.show()

train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]  # como fracción del train set

# Calcular learning curve
train_sizes_abs, train_scores, test_scores = learning_curve(
    modelo,                   # tu modelo ya entrenado
    X_train,
    y_train,
    cv=5,
    train_sizes=train_sizes,
    scoring='accuracy',       # puedes cambiar a 'precision_weighted' si quieres
    n_jobs=-1
)

# Promedio y desviación estándar
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

# Graficar
plt.figure(figsize=(8,6))
plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training score')
plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)

plt.plot(train_sizes_abs, test_mean, 'o-', color='green', label='Cross-validation score')
plt.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, color='green', alpha=0.1)

plt.title('Learning Curve - Decision Tree')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()
