import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import time 
df = pd.read_csv("dataset_ordenado.csv")
df = shuffle(df, random_state=42)
test_idx = []
train_idx = []
Y = df['category']
n_test_por_clase = min(21, Y.value_counts().min())
for clase in df['category'].unique():
    indices_clase = df[df['category'] == clase].index
    test_indices_clase = indices_clase[:n_test_por_clase]
    train_indices_clase = indices_clase[n_test_por_clase:]
    test_idx.extend(test_indices_clase)
    train_idx.extend(train_indices_clase)
X_train = df.loc[train_idx].drop(columns=['category'])
Y_train = df.loc[train_idx, 'category']
X_test  = df.loc[test_idx].drop(columns=['category'])
Y_test  = df.loc[test_idx, 'category']
gb = GradientBoostingClassifier(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.5,0.1],
    "max_depth": [2, 3, 5, 8],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.6, 0.8, 1.0],   # fracción de muestras por árbol
    "max_features": [None, "sqrt", "log2"]
}
grid_search = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)
inicio=time.time()
grid_search.fit(X_train, Y_train)
fin=time.time()

print("⏱ Tiempo total de GridSearch: {:.2f} segundos".format(fin - inicio))
print("Mejores parámetros:", grid_search.best_params_)
print("Mejor score CV:", grid_search.best_score_)
best_model = grid_search.best_estimator_
test_acc = best_model.score(X_test, Y_test)
print("Accuracy en test:", test_acc)
