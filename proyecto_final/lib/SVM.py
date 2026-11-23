from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from lib.base import BaseModel
from utils.metrics import evaluate_model
from utils.plots import plot_confusion_matrix,plot_multiclass_roc
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
class SVM_kernel(BaseModel):
    def __init__(self,params) -> None:
        super().__init__(params["model"])
        self.params=params
        self.build()

    def build(self):
        model_params={k:v for k,v in self.params.items() if k!="model"}
        model_params["probability"]=True
        self.model=SVC(**model_params)
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    def predict(self,X):
        return self.model.predict(X)
    def evaluate(self, X_test, y_test,name=""):
        y_pred=self.predict(X_test)
        y_prob=self.model.predict_proba(X_test)
        self.metrics=evaluate_model(y_test,y_pred,y_prob,self.name)
        cm=self.metrics["confusion_matrix"]
        classes=np.unique(y_test)
        plot_confusion_matrix(cm,classes,self.name+name,"plots/confussion")
        plot_multiclass_roc(y_test, y_prob, classes, self.name + name, "plots/ROC")
class SVM_Lineal(BaseModel):
    def __init__(self):
        super().__init__("SVM_Lineal")
        self.build()
    def build(self):
        self.model=LinearSVC(max_iter=2000,C=1)

        self.model=CalibratedClassifierCV(self.model,cv=5,method="sigmoid")
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.model.fit(X_train,y_train)
    def predict(self,X):
        return self.model.predict(X)
    def evaluate(self, X_test, y_test,name=""):
        y_pred=self.predict(X_test)
        y_prob=self.model.predict_proba(X_test)
        self.metrics=evaluate_model(y_test,y_pred,y_prob,self.name)
        cm=self.metrics["confusion_matrix"]
        classes=np.unique(y_test)
        plot_confusion_matrix(cm,classes,self.name+name,"plots/confussion")
        plot_multiclass_roc(y_test, y_prob, classes, self.name + name, "plots/ROC")
