from sklearn.ensemble import RandomForestClassifier
from package import reader_csv,find_hyperparamters,create_file
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import balanced_accuracy_score, make_scorer,f1_score

dataframe=reader_csv("features_dataset_completo.csv").drop(columns=["fpfh_frac40","cz_frac50","cz_frac40","fpfh_frac50","cz_frac20","dy_frac50","dx_frac50","fpfh_frac30","dx_frac30","dy_frac20","dx_frac40","cy_frac50","dz_frac40","dy_frac30","cy_frac40","cx_frac40","cx_frac50","cy_frac30","dz_frac30","cx_frac20","dz_frac20","curvature_frac40","compacidad_frac30","compacidad_frac40","compacidad_frac20","curvature_frac30","border_frac40","border_frac50"])
X = dataframe.drop(columns='category')
y = dataframe["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
shuffle_dataset=(X_train,y_train,X_test,y_test)
class RFObjective:
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __call__(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, ),  
            "max_depth": trial.suggest_int("max_depth", 5, 10),                    
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10,step=1),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10,step=1),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0)
        }
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )
        model = self.build_model(params)
        scoring = make_scorer(balanced_accuracy_score)
        scores = cross_val_score(model, self.X, self.y, cv=5, scoring=scoring, n_jobs=-1)
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)

        # F1 macro, evitando NaN con zero_division=0
        score = f1_score(y_val, model.predict(X_val), average="macro", zero_division=0)
        return score
    def build_model(self, params):
        return RandomForestClassifier(
            **params, random_state=42, n_jobs=-1
        )
objective = RFObjective(X_train, y_train)
tiempo, best_params, best_score, best_model, test_acc, study = find_hyperparamters(objective, shuffle_dataset, n_trials=50)
print(tiempo)
print(best_params)
print(best_score)
print(best_model)
print(test_acc)
df_results = study.trials_dataframe()
#create_file(df_results, "random_forest")
