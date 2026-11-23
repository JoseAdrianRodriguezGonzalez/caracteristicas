import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, classification_report
df=pd.read_csv('dataset_ordenado.csv')
X=df.drop(columns='category')
Y=df['category']
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42,stratify=Y)
modelo=DecisionTreeClassifier(random_state=42,max_depth=5,min_samples_split=5,
                              ccp_alpha=0.005)
modelo.fit(X_train,y_train)
y_pred=modelo.predict(X_test)
accuray=accuracy_score(y_test,y_pred)
train_sizes, train_scores, test_scores = learning_curve(
    estimator=modelo,
    X=X_train,  # Usar SOLO el training set
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),  # De 10% a 100%
    cv=5,                                   # 5-fold cross validation
    scoring='accuracy',
    n_jobs=-1,                              # Usar todos los cores
    random_state=42
)

# 5. Calcular medias y desviaciones
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# 6. Graficar curva de aprendizaje
plt.figure(figsize=(12, 8))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Entrenamiento', linewidth=2, markersize=8)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, 's-', color='red', label='Validación Cruzada', linewidth=2, markersize=8)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='red')
final_test_accuracy = accuracy_score(y_test, y_pred)
plt.axhline(y=final_test_accuracy, color='green', linestyle='--', 
            label=f'Test Final = {final_test_accuracy:.3f}')

plt.title('Curva de Aprendizaje del Modelo Final\n(Degree Tree - max_depth=5, min_samples_split=5)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Tamaño del Conjunto de Entrenamiento', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.05)  # Ajustar para mejor visualización

# Añadir anotaciones
plt.annotate(f'Accuracy final en test: {final_test_accuracy:.3f}',
             xy=(train_sizes[-1], final_test_accuracy),
             xytext=(train_sizes[5], final_test_accuracy - 0.1),
             arrowprops=dict(arrowstyle='->', color='green'),
             fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('curva_aprendizaje_modelo_final.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Mostrar métricas numéricas
print("📊 MÉTRICAS DE LA CURVA DE APRENDIZAJE:")
print(f"{'Tamaño':>8} {'Train Mean':>10} {'Train Std':>10} {'Test Mean':>10} {'Test Std':>10}")
print("-" * 60)
for i, size in enumerate(train_sizes):
    print(f"{size:8.0f} {train_mean[i]:10.4f} {train_std[i]:10.4f} {test_mean[i]:10.4f} {test_std[i]:10.4f}")

# 8. Entrenar y evaluar el modelo final (igual que tu código)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 RESULTADO FINAL EN TEST SET:")
print(f"Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 9. Análisis de la curva
brecha_final = train_mean[-1] - test_mean[-1]
print(f"\n🔍 ANÁLISIS DE LA CURVA:")
print(f"Brecha final train vs validation: {brecha_final:.4f}")
if brecha_final < 0.05:
    print("✅ BUEN AJUSTE: Brecha pequeña indica buen balance")
elif brecha_final < 0.1:
    print("⚠️  AJUSTE MODERADO: Brecha aceptable")
else:
    print("❌ OVERFITTING: Brecha grande indica sobreajuste")