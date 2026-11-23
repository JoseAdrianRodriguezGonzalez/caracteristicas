scripts=("random_forest.py" "adaboost.py" "gradient_boost.py")
for script in "${scripts[@]}"; do
    base=$(basename "$script" .py)
    echo "🚀 Ejecutando $script..."

    python "$script">"${base}.txt"

    echo "✅ Salida guardada en ${base}.txt"
    echo "---------------------------------"
done