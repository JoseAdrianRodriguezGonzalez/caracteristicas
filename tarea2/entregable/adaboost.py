from package import reader_csv,find_hyperparamters,create_file
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import numpy as np
from sklearn.metrics import balanced_accuracy_score, make_scorer
dataframe=reader_csv("FEAT_PC_objects.csv")
X = dataframe.drop(columns='category')
y = dataframe["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
shuffle_dataset=(X_train,y_train,X_test,y_test)

class AdaObjective:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 2.0, log=True),
            "algorithm": trial.suggest_categorical("algorithm", ["SAMME"]),
            "base_max_depth": trial.suggest_int("base_max_depth", 1, 5),
            "base_min_samples_split": trial.suggest_int("base_min_samples_split", 2, 20),
            "base_min_samples_leaf": trial.suggest_int("base_min_samples_leaf", 1, 10)
        }
        model = self.build_model(params)
        scoring = make_scorer(balanced_accuracy_score)
        scores = cross_val_score(model, self.X, self.y, cv=5, scoring=scoring, n_jobs=-1)
        scores = cross_val_score(model, self.X, self.y, cv=5, scoring="accuracy", n_jobs=-1)
        return np.mean(scores)

    def build_model(self, params):
        base_max_depth = params.pop("base_max_depth")
        base_min_samples_split = params.pop("base_min_samples_split")
        base_min_samples_leaf = params.pop("base_min_samples_leaf")
        base_tree = DecisionTreeClassifier(
            max_depth=base_max_depth,
            min_samples_split=base_min_samples_split,
            min_samples_leaf=base_min_samples_leaf,
            random_state=42
        )
        return AdaBoostClassifier(
            estimator=base_tree,   
            **params,
            random_state=42
        )

objective = AdaObjective(X_train, y_train)
tiempo, best_params, best_score, best_model, test_acc, study = find_hyperparamters(objective, shuffle_dataset, n_trials=50)
print(tiempo)
print(best_params)
print(best_score)
print(best_model)
print(test_acc)
df_results = study.trials_dataframe()
#create_file(df_results, "random_forest")
