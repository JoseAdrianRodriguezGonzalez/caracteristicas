from lib.base import BaseModel
from utils.metrics import evaluate_model
from utils.plots import plot_confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

class LDA_Model(BaseModel):
    def __init__(self, solver="lsqr"):
        super().__init__("LDA")
        self.solver = solver
        self.build()

    def build(self):
        self.model = LinearDiscriminantAnalysis(solver=self.solver)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def reduce_dimensionality(self,X,threshold=0.9):
        if not hasattr(self.model,"explained_variance_ratio_"):
            raise ValueError("Debes entrenar el modelo antes de reducir")
        var_ratio=self.model.explained_variance_ratio_
        cumulative=np.cumsum(var_ratio)
        n_components=np.searchsorted(cumulative,threshold)+1
        X_proj=self.model.transform(X)[:,:n_components]
        return X_proj,n_components
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X_test, y_test, name=""):
        y_pred = self.predict(X_test)
        self.metrics = evaluate_model(y_test, y_pred, self.name)
        cm = self.metrics["confusion_matrix"]
        classes = np.unique(y_test)
        plot_confusion_matrix(cm, classes, self.name + name, "plots/")
    def classify_reduced(self, X_train, X_test, y_train, y_test, threshold=0.9,name=""):
        self.train(X_train, y_train)
        X_train_red, n_comp = self.reduce_dimensionality(X_train, threshold)
        X_test_red, _ = self.reduce_dimensionality(X_test, threshold)
#Clasificacion LDA
        lda_clf=LinearDiscriminantAnalysis(solver=self.solver)
        lda_clf.fit(X_train_red,y_train)
        y_pred=lda_clf.predict(X_test_red)

        self.metrics = evaluate_model(y_test, y_pred, f"{self.name}_{n_comp}comp")
        cm = self.metrics["confusion_matrix"]
        classes = np.unique(y_test)
        plot_confusion_matrix(cm, classes, f"{self.name}_{name}_{n_comp}comp", "plots/")

        print(f"Usando {n_comp} componentes para {threshold*100:.1f}% de varianza.")
        return X_train_red, X_test_red, y_pred
