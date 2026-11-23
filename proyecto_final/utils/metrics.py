from sklearn.metrics import roc_auc_score,confusion_matrix, accuracy_score,f1_score,recall_score,precision_score
def evaluate_model(y_true,y_pred,y_prob,model_name="name"):
    acc=accuracy_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred,average="macro")
    recall=recall_score(y_true,y_pred,average="macro")
    precision=precision_score(y_true,y_pred,average="macro")
    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm
    }
    if y_prob is not None:
        try:
            auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            auc_micro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="micro")
            metrics["auc_macro"] = auc_macro
            metrics["auc_micro"] = auc_micro
            print(f"[{model_name}] AUC_macro: {auc_macro:.4f} | AUC_micro: {auc_micro:.4f}")
        except Exception as e:
            print("[WARN] No se pudo calcular AUC multiclase:", e)
    print(f" [{model_name}] Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    return metrics  

