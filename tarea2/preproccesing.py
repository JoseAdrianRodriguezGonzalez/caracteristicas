from package import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Cargar datos
#df = reader_csv("features_dataset_completo.csv").drop(columns=["fpfh_frac40","cz_frac50","cz_frac40","fpfh_frac50","cz_frac20"])
#df = pd.read_csv("features_dataset_completo.csv").drop(columns=["fpfh_frac40","cz_frac50","cz_frac40","fpfh_frac50","cz_frac20","dy_frac50","dx_frac50","fpfh_frac30","dx_frac30","dy_frac20","dx_frac40","cy_frac50"])
#df = pd.read_csv("features_dataset_completo.csv").drop(columns=["fpfh_frac40","cz_frac50","cz_frac40","fpfh_frac50","cz_frac20","dy_frac50","dx_frac50","fpfh_frac30","dx_frac30","dy_frac20","dx_frac40","cy_frac50"])
df = pd.read_csv("features_dataset_completo.csv").drop(columns=["fpfh_frac40","cz_frac50","cz_frac40","fpfh_frac50","cz_frac20","dy_frac50","dx_frac50","fpfh_frac30","dx_frac30","dy_frac20","dx_frac40","cy_frac50","dz_frac40","dy_frac30","cy_frac40","cx_frac40","cx_frac50","cy_frac30","dz_frac30","cx_frac20","dz_frac20"])

#df = pd.read_csv("features_dataset_completo.csv").drop(columns=["fpfh_frac40","cz_frac50","cz_frac40","fpfh_frac50","cz_frac20","dy_frac50","dx_frac50","fpfh_frac30","dx_frac30","dy_frac20","dx_frac40","cy_frac50","dz_frac40","dy_frac30","cy_frac40","cx_frac40","cx_frac50","cy_frac30","dz_frac30","cx_frac20","dz_frac20","curvature_frac40","compacidad_frac30","compacidad_frac40","compacidad_frac20","curvature_frac30"])

#df = pd.read_csv("features_dataset_completo.csv").drop(columns=["fpfh_frac40","cz_frac50","cz_frac40","fpfh_frac50","cz_frac20","dy_frac50","dx_frac50","fpfh_frac30","dx_frac30","dy_frac20","dx_frac40","cy_frac50","dz_frac40","dy_frac30","cy_frac40","cx_frac40","cx_frac50","cy_frac30","dz_frac30","cx_frac20","dz_frac20","curvature_frac40","compacidad_frac30","compacidad_frac40","compacidad_frac20","curvature_frac30"])
#df = pd.read_csv("features_dataset_completo.csv").drop(columns=["fpfh_frac40","cz_frac50","cz_frac40","fpfh_frac50","cz_frac20","dy_frac50","dx_frac50","fpfh_frac30","dx_frac30","dy_frac20","dx_frac40","cy_frac50","dz_frac40","dy_frac30","cy_frac40","cx_frac40","cx_frac50","cy_frac30","dz_frac30","cx_frac20","dz_frac20","curvature_frac40","compacidad_frac30","compacidad_frac40","compacidad_frac20","curvature_frac30","border_frac40","border_frac50"])
X = df.drop(columns='category')
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# División y entrenamiento del modelo
shuffle_dataset=(X_train,y_train,X_test,y_test)
param_grid = [
    {"C": [0.001, 0.01, 0.1, 1, 10, 50, 100],
     "solver": ["lbfgs"],
     "penalty": ["l2"],
     "max_iter": [1000]},
    {"C": [0.001, 0.01, 0.1, 1, 10, 50, 100],
     "solver": ["saga"],
     "penalty": ["l1", "l2"],
     "max_iter": [1000]}                       
]

grid_search_parameters = {
    "cv": 5,
    "n_jobs": -1,
    "verbose": 0,
    "scoring": "accuracy"
}

model = LogisticRegression(multi_class="multinomial", random_state=42)
t, a,c, mejor, d = find_hyperparamters_(param_grid, model, grid_search_parameters, shuffle_dataset)
print(t)
print(a)
print(c)
print(d)
acc = evaluate_model(mejor, shuffle_dataset)
confussion_matrix(mejor, shuffle_dataset, "logistic_regression")

params = {
    "cv": 5,
    "n_jobs": -1,
    "scoring": "accuracy",
    "train_sizes": np.linspace(0.3, 1.0, 10)
}
learning_curve_(shuffle_dataset, mejor, params, "logistic_regression")
print(t)
print(a)
print(c)
print(d)
print(acc)