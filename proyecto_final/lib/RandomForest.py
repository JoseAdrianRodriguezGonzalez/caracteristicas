
# lib/RF.py
from sklearn.ensemble import RandomForestClassifier
from lib.base import BaseModel
import numpy as np
from utils.metrics import evaluate_model
from utils.plots import plot_confusion_matrix, plot_multiclass_roc


class RandomForest(BaseModel):
    def __init__(self, **params):
        """
        params puede venir como:
        {
            "model": "RandomForest",
            "n_estimators": 200,
            "max_depth": 10,
            ...
        }
        """
        model_name = params.get("model", "RandomForest")
        super().__init__(model_name)
        self.params = params
        self.build()

    def build(self):
        model_params = {k: v for k, v in self.params.items() if k != "model"}
        self.model = RandomForestClassifier(
            **model_params,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, name=""):
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)

        self.metrics = evaluate_model(y_test, y_pred, y_prob, self.name)

        cm = self.metrics["confusion_matrix"]
        classes = np.unique(y_test)

        plot_confusion_matrix(cm, classes, self.name + name, "plots/confussion")
        plot_multiclass_roc(y_test, y_prob, classes, self.name + name, "plots/ROC")

