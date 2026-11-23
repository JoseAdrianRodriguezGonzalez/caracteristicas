import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
from time import time
def reader_csv(name_file=""):
    """_summary_:
    Reads a csv and return a dataframe from it.
    Args:
        name_file (str, optional): Write down the csv's name,  Defaults to "".
    Returns:
        dataframe: dataframe read by the csv
    """
    return pd.read_csv(name_file)
def split_categories(df,y_category=''):
    """_summary_
    Receives a dataframe and split it out according to the y_category.
    Args:
        df (dataframe): reads a dataframe and split it out as X and Y
    Return
        tuple: the first element consist on the X variable, and the second the Y variable
    """
    return df.drop(columns=y_category),df[y_category]
def train_test_split_custom(dataframe,y_category,n_test=21):
    """_summary_
    Shuffle the samples and splits off the dataset as X and y to train and the same process with test
    Args:
        dataframe (Dataframe): It corresponds to a dataframe 
        y_category (string): name of the y category
        n_test (int, optional): The minimum number of samples. Defaults to 21.
    Returns:
        X_train (pd.dataFrame): predictory variables for training
        y_train (pd.dataFrame): train labels
        X_train (pd.dataFrame): predictory variables for test
        y_test (pd.dataFrame): test labels
    """
    data=shuffle(dataframe,random_state=42)
    test_indices,train_indices = [],[]
    n_test_por_clase = min(n_test, data[y_category].value_counts().min())
    for clase in data[y_category].unique():
        indices_clase = data[data[y_category] == clase].index
        test_indices_clase = indices_clase[:n_test_por_clase] 
        train_indices_clase = indices_clase[n_test_por_clase:]
        test_indices.extend(test_indices_clase)
        train_indices.extend(train_indices_clase)
    X_train = data.loc[train_indices].drop(columns=[y_category])
    y_train = data.loc[train_indices][y_category]
    X_test = data.loc[test_indices].drop(columns=[y_category])
    y_test = data.loc[test_indices][y_category]
    return X_train,y_train,X_test,y_test
def find_hyperparamters(objective_func, dataset, n_trials=50):
    X_train, y_train, X_test, y_test = dataset

    study = optuna.create_study(direction="maximize")
    start = time()
    study.optimize(objective_func, n_trials=n_trials)
    end = time()

    best_params = study.best_trial.params
    best_score = study.best_value

    # Reentrenar modelo con mejores parámetros
    best_model = objective_func.build_model(best_params)
    best_model.fit(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)

    print("⏱ Tiempo total Optuna: {:.2f} segundos".format(end - start))
    print("Mejores parámetros:", best_params)
    print("Mejor score CV:", best_score)
    print("Accuracy en test:", test_acc)

    return (end - start), best_params, best_score, best_model, test_acc, study
def create_folder(name):
    os.makedirs(name,exist_ok=True)
def create_file(elements,name):
    create_folder(name)
    directory_name=name+"/"+name+".txt"
    with open(directory_name,"a",encoding="utf-8") as f:
        for element in elements:
            f.write(f"{element}\n")
def read_file(name):
    directory_name=''
    elements=[]
    if '/' not in name:
        directory_name=name+"/"+name+".txt"
    else:
        directory_name=name
    with open(directory_name,'rt') as file:
        elements=[f for f in file]
    return elements
def evaluate_model(model,data_shuffled):
    model.fit(data_shuffled[0],data_shuffled[1])
    y_pred=model.predict(data_shuffled[2])
    return y_pred
def confussion_matrix(model,data_shuffled,name):
    y_pred=evaluate_model(model,data_shuffled)
    cm=confusion_matrix(data_shuffled[3],y_pred,labels=model.classes_)
    disp =ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de confusion {name}")
    os.makedirs(name,exist_ok=True)
    plt.savefig(f"{name}/confusion_matrix_{name}.png", dpi=300, bbox_inches="tight")
    
    plt.show()
def learning_curve_(data_shuffled,model,params,name):
    train_size,train_score, test_scores=learning_curve(
    model,data_shuffled[0],data_shuffled[1],
    **params)
    train_mean=np.mean(train_score,axis=1)
    train_std = np.std(train_score,axis=1)
    test_mean=np.mean(test_scores,axis=1)
    test_std = np.std(test_scores,axis=1)

    plt.plot(train_size, train_mean, 'o-', color="red", label="Entrenamiento", linewidth=2)
    plt.plot(train_size, test_mean, 'o-', color="green", label="Validación", linewidth=2)


    plt.fill_between(train_size, train_mean-train_std, train_mean+train_std, alpha=0.1, color="r")
    plt.fill_between(train_size, test_mean-test_std, test_mean+test_std, alpha=0.1, color="g")

    plt.title(f"Curva de Aprendizaje {name}")
    plt.xlabel("Número de muestras de entrenamiento")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    os.makedirs(name,exist_ok=True)
    plt.savefig(f"{name}/learning_curve_{name}.png", dpi=300, bbox_inches="tight")
  
    plt.show()

def find_hyperparamters_(parameters,model,grid_search_parameters,dataset):
    grid_search=GridSearchCV(
        param_grid=parameters,
        estimator=model,
        **grid_search_parameters
    )
    start=time()
    grid_search.fit(dataset[0],dataset[1])
    end=time()
    print("⏱ Tiempo total de GridSearch: {:.2f} segundos".format(end - start))
    print("Mejores parámetros:", grid_search.best_params_)
    print("Mejor score CV:", grid_search.best_score_)
    mejor_modelo = grid_search.best_estimator_
    test_acc = mejor_modelo.score(dataset[2], dataset[3])
    print("Accuracy en test:", test_acc)
    return (end - start),grid_search.best_params_,grid_search.best_score_,mejor_modelo,test_acc