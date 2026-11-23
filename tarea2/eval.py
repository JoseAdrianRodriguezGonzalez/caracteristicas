from package import read_file,reader_csv,train_test_split_custom,evaluate_model,confussion_matrix,learning_curve_
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
ada_elements=read_file('Adaboost')
grad_elements=read_file('gradient_boost')
rnd_elements=read_file('random_forest')
grad_object="".join([grad_elements[3],grad_elements[4]]).strip()
ada_object="".join([ada_elements[3],ada_elements[4],ada_elements[5]]).strip()
rnd_object=rnd_elements[3]
models=[eval(rnd_object),eval(ada_object),eval(grad_object)]
names=['random_forest','Adaboost','gradient_boost']
dataframe=reader_csv("features_dataset_completo.csv").drop(columns=["fpfh_frac40","cz_frac50","cz_frac40","fpfh_frac50","cz_frac20","dy_frac50","dx_frac50","fpfh_frac30","dx_frac30","dy_frac20","dx_frac40","cy_frac50","dz_frac40","dy_frac30","cy_frac40","cx_frac40","cx_frac50","cy_frac30","dz_frac30","cx_frac20","dz_frac20","curvature_frac40","compacidad_frac30","compacidad_frac40","compacidad_frac20","curvature_frac30","border_frac40","border_frac50"])
X = dataframe.drop(columns='category')
y = dataframe["category"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)
dataframe_shuffle=(X_train,y_train, X_test,  y_test)
for model,name in zip(models,names):
    params = {
        "cv": 5,
        "n_jobs": -1,
        "scoring": "accuracy",
        "train_sizes": np.linspace(0.3, 1.0, 10)
    }

    evaluate_model(model,dataframe_shuffle)
    confussion_matrix(model,dataframe_shuffle,name)
    learning_curve_(dataframe_shuffle,model,
                    params,name)
    
