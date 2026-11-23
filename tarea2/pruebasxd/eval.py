from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
df=pd.read_csv("dataset_ordenado.csv")
X=df.drop(columns='category')
Y=df['category']

data=shuffle(df,random_state=42)
test_idx=[]
train_idx=[]
n_test_por_clase=min(21,Y.value_counts().min())
for clase in df['category'].unique():
    indices_clase=df[df['category']==clase].index
    test_indices_clase=indices_clase[:n_test_por_clase]
    train_indices_clase= indices_clase[n_test_por_clase:]
    test_idx.extend(test_indices_clase)
    train_idx.extend(train_indices_clase)
X_train=df.loc[train_idx].drop(columns=['category'])
Y_train=df.loc[train_idx]['category']
X_test= df.loc[test_idx].drop(columns=['category'])
Y_test=df.loc[test_idx]['category']
modelo=RandomForestClassifier(random_state=42,
                              bootstrap=True,max_depth=None,
                              min_samples_leaf=1,
                              min_samples_split=2,
                              n_estimators=100)
modelo.fit(X_train,Y_train)
y_pred=modelo.predict(X_test)
cm=confusion_matrix(Y_test,y_pred,labels=modelo.classes_)
disp =ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=modelo.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de confusion random forest")
plt.show()

train_size,train_score, test_scores=learning_curve(
    modelo,X_train,Y_train,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    train_sizes=np.linspace(0.1,1.0,10)
)

train_mean=np.mean(train_score,axis=1)
train_std = np.std(train_score,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)

plt.plot(train_size, train_mean, 'o-', color="red", label="Entrenamiento", linewidth=2)
plt.plot(train_size, test_mean, 'o-', color="green", label="Validación", linewidth=2)


plt.fill_between(train_size, train_mean-train_std, train_mean+train_std, alpha=0.1, color="r")
plt.fill_between(train_size, test_mean-test_std, test_mean+test_std, alpha=0.1, color="g")

plt.title("Curva de Aprendizaje Random Forest")
plt.xlabel("Número de muestras de entrenamiento")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.show()
