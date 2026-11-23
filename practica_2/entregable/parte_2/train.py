import torch 

class Trainer:
    def __init__(self,model,train_loader,val_loader,device,criterion,optimizer):
        self.model=model.to(device)
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.device=device
        self.criterion=criterion
        self.optimizer=optimizer
        self.history={"train_acc":[],"val_acc":[],"train_loss":[],"val_loss":[]}
    def train(self):
        self.model.train()
        total_loss=0
        correct=0
        total=0
        for imgs,labels in self.train_loader:
            imgs,labels=imgs.to(self.device),labels.to(self.device)
            self.optimizer.zero_grad()
            outputs=self.model(imgs)
            loss=self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            total_loss+=loss.item()
            _,preds=outputs.max(1)
            total+=labels.size(0)
            correct+=(preds==labels).sum().item()
        return total_loss /len(self.train_loader),correct/total
    def evaluate(self):
        self.model.eval()
        correct,total=0,0
        total_loss=0.0
        with torch.no_grad():
            for imgs,labels in self.val_loader:
                imgs,labels=imgs.to(self.device),labels.to(self.device)
                outputs=self.model(imgs)
                loss=self.criterion(outputs,labels)
                total_loss+=loss.item()
                _,preds=outputs.max(1)
                total+=labels.size(0)
                correct+=(preds==labels).sum().item()
        return total_loss/len(self.val_loader),correct/total
    def fit(self,epochs=10,patience=3):
        best_val_loss=float("inf")
        best_model_state=None
        epochs_no_improve=0
        for epoch in range(epochs):
            train_loss,train_acc=self.train()
            val_loss,val_acc=self.evaluate()
            print(f"Epoca [{epoch+1}/{epochs}] | Train Acc: {train_acc:.4f} "
                f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
            self.history["train_acc"].append(train_acc)
            self.history["train_loss"].append(train_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_loss"].append(val_loss)
            if val_loss<best_val_loss:
                best_val_loss=val_loss
                best_model_state=self.model.state_dict()
                epochs_no_improve=0
            else:
                epochs_no_improve+=1
            if epochs_no_improve>=patience:
                print(f"Early stopping activado despues de {epoch+1} epocas")
            
        if best_model_state:
            self.model.load_state_dict(best_model_state)
