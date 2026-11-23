from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,recall_score,precision_score
def evaluate_model(y_true,y_pred,model_name="name"):
    acc=accuracy_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred,average="macro")
    recall=recall_score(y_true,y_pred,average="macro")
    precision=precision_score(y_true,y_pred,average="macro")
    cm = confusion_matrix(y_true, y_pred)
    print(f" [{model_name}] Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall ,
        "f1": f1,
        "confusion_matrix": cm
    }

