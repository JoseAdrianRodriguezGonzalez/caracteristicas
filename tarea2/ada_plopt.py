from sklearn.model_selection import train_test_split
from package import reader_csv,evaluate_model,confussion_matrix,learning_curve_
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score
dataframe=reader_csv("features_dataset_completo.csv").drop(columns=["fpfh_frac40","cz_frac50","cz_frac40","fpfh_frac50","cz_frac20",])
X = dataframe.drop(columns='category')
y = dataframe["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
dataframe_shuffle=(X_train,y_train, X_test,  y_test)
model=AdaBoostClassifier(algorithm='SAMME',
                   estimator=DecisionTreeClassifier(max_depth=4,
                                                    min_samples_leaf=6,
                                                    min_samples_split=8,
                                                    random_state=42),
                   learning_rate=0.005, random_state=42)
params = {
    "cv": 5,
    "n_jobs": -1,
    "scoring": "accuracy",
    "train_sizes": np.linspace(0.3, 1.0, 10)
}

evaluate_model(model,dataframe_shuffle)
#confussion_matrix(model,dataframe_shuffle,"Adaboost")
#learning_curve_(dataframe_shuffle,model,
#                params,"Adaboost")


y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

print("=== Training Metrics ===")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Balanced Accuracy:", balanced_accuracy_score(y_train, y_pred_train))
print("F1 Macro:", f1_score(y_train, y_pred_train, average="macro"))
print("Precision Macro:", precision_score(y_train, y_pred_train, average="macro"))
print("Recall Macro:", recall_score(y_train, y_pred_train, average="macro"))

print("\n=== Test Metrics ===")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred_test))
print("F1 Macro:", f1_score(y_test, y_pred_test, average="macro"))
print("Precision Macro:", precision_score(y_test, y_pred_test, average="macro"))
print("Recall Macro:", recall_score(y_test, y_pred_test, average="macro"))