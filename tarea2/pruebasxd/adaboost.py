import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import time
df=pd.read_csv('dataset_ordenado.csv')
X=df.drop(columns='category')
Y=df['category']
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

modelo=DecisionTreeClassifier(random_state=42)
ada=AdaBoostClassifier(estimator=modelo,random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],        
    "learning_rate": [0.01,0.05 ,0.1, ],     
    "estimator__max_depth": [1, 2, 3],     
    "estimator__min_samples_split": [2, 5, 10]  
}
grid_search = GridSearchCV(
    estimator=ada,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)
inicio=time.time()
grid_search.fit(X_train,y_train)
fin=time.time()
print("⏱ Tiempo total de GridSearch: {:.2f} segundos".format(fin - inicio))
print("Mejores parámetros:", grid_search.best_params_)
print("Mejor score CV:", grid_search.best_score_)
mejor_modelo = grid_search.best_estimator_
test_acc = mejor_modelo.score(X_test, y_test)
print("Accuracy en test:", test_acc)