from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity
from lib.base import BaseModel
from utils.metrics import evaluate_model
from utils.plots import plot_confusion_matrix,plot_multiclass_roc
import numpy as np 
from sklearn.base import BaseEstimator, ClassifierMixin


class GaussianNaiveBayes(BaseModel):
    def __init__(self) :
        super().__init__("NaiveBayesGaussian")
        self.build()
         
    def build(self):
        self.model=GaussianNB()
    def train(self,X_train,y_train):
        self.model.fit(X_train,y_train)
    def predict(self,X):
        return self.model.predict(X)
    def evaluate(self, X_test, y_test,name=""):
        y_pred=self.predict(X_test)
        y_prob=self.model.predict_proba(X_test)
        self.metrics=evaluate_model(y_test,y_pred,y_prob,"NaiveBayesGaussian") 
        cm=self.metrics["confusion_matrix"]
        classes=np.unique(y_test)
        plot_confusion_matrix(cm,classes,self.name+name,"plots/confussion")
        plot_multiclass_roc(y_test,y_prob,classes,self.name+name,"plots/ROC")
class KernelNaiveBayes(BaseModel, BaseEstimator, ClassifierMixin):
    """
    Modelo Naive Bayes con estimación de densidad por kernel (KDE).
    Compatible con la API de scikit-learn (fit, predict, score).
    """
    def __init__(self, bandwidth=1.0):
        super().__init__("NaiveBayesKernel")
        self.bandwidth = bandwidth
        self.kdes = {}
        self.class_priors = {}

    # === Métodos principales ===
    def build(self):
        """Mantiene compatibilidad con tu arquitectura BaseModel."""
        self.kdes = {}
        self.class_priors = {}

    def fit(self, X, y):
        """
        Entrena el modelo (alias de train).
        Requerido por scikit-learn para funcionar con learning_curve.
        """
        self.train(X, y)
        return self

    def train(self, X_train, y_train):
        """Entrena el modelo KDE por clase."""
        self.classes = np.unique(y_train)
        for c in self.classes:
            X_c = X_train[y_train == c]
            kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth,algorithm="ball_tree").fit(X_c)
            self.kdes[c] = kde
            self.class_priors[c] = len(X_c) / len(X_train)

    def predict(self, X):
        """Predice etiquetas de clase según máxima probabilidad a posteriori."""
        log_probs = []
        for c in self.classes:
            log_likelihood = self.kdes[c].score_samples(X)
            log_prior = np.log(self.class_priors[c])
            log_probs.append(log_likelihood + log_prior)
        log_probs = np.array(log_probs).T
        return self.classes[np.argmax(log_probs, axis=1)]

    def evaluate(self, X_test, y_test,name=""):
        """Evalúa con tus métricas y genera matriz de confusión."""
        y_pred = self.predict(X_test)
        y_prob=self.predict_proba(X_test)
        self.metrics = evaluate_model(y_test, y_pred,y_prob, "NaiveBayesKernel")
        cm = self.metrics["confusion_matrix"]
        classes = np.unique(y_test) 
        plot_confusion_matrix(cm, classes, self.name+name, "plots/confussion")
        plot_multiclass_roc(y_test,y_prob,classes,self.name+name,"plots/ROC")
    def predict_proba(self, X):
        """
        Retorna probabilidades por clase (no solo scores).
        Necesario para ROC/AUC.
        """
        log_probs = []
        for c in self.classes:
            log_likelihood = self.kdes[c].score_samples(X)
            log_prior = np.log(self.class_priors[c])
            log_probs.append(log_likelihood + log_prior)

        log_probs = np.vstack(log_probs).T  # shape = (n_samples, n_classes)

        # Convertir log-probabilidades a probabilidades reales normalizadas
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def score(self, X, y):
        """
        Devuelve accuracy — requerido por learning_curve y cross_val_score.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
