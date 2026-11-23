from sklearn.ensemble import GradientBoostingClassifier
from package import reader_csv,find_hyperparamters,create_file
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import balanced_accuracy_score, make_scorer
import numpy as np
dataframe=reader_csv("FEAT_PC_objects.csv")
X = dataframe.drop(columns='category')
y = dataframe["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

shuffle_dataset=(X_train,y_train,X_test,y_test)
class GBObjective:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "loss": trial.suggest_categorical("loss", ['log_loss']),
            "criterion": trial.suggest_categorical("criterion", ["friedman_mse", "squared_error"])
        }
        model = self.build_model(params)
        scoring = make_scorer(balanced_accuracy_score)
        scores = cross_val_score(model, self.X, self.y, cv=5, scoring=scoring, n_jobs=-1)
        return np.mean(scores)

    def build_model(self, params):
        return GradientBoostingClassifier(
            **params, random_state=42
        )
objective = GBObjective(X_train, y_train)
tiempo, best_params, best_score, best_model, test_acc, study = find_hyperparamters(objective, shuffle_dataset, n_trials=50)
print(tiempo)
print(best_params)
print(best_score)
print(best_model)
print(test_acc)
df_results = study.trials_dataframe()
#create_file(df_results, "random_forest")
