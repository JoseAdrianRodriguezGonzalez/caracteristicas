from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import pandas as pd
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
modelo=RandomForestClassifier(random_state=42)
param_grid={
    "n_estimators":[50,70,100,200,500],
    "max_depth":[None,5,10,20],
    "min_samples_split":[2,5,7,10],
    "min_samples_leaf":[1,2,3,4],
    "bootstrap":[True,False]
}
grid_search=GridSearchCV(
    estimator=modelo,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=0,
    scoring="accuracy"
)
inicio = time.time()
grid_search.fit(X_train, Y_train)
fin = time.time()
print("⏱ Tiempo total de GridSearch: {:.2f} segundos".format(fin - inicio))
print("Mejores parámetros:", grid_search.best_params_)
print("Mejor score CV:", grid_search.best_score_)
mejor_modelo = grid_search.best_estimator_
test_acc = mejor_modelo.score(X_test, Y_test)
print("Accuracy en test:", test_acc)