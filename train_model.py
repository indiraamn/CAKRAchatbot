import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("dataset.csv")

df["Motivasi"] = df["Motivasi"].map({
    "Rendah": 0,
    "Sedang": 1,                
    "Tinggi": 2
})

fitur = [
    "Semester",
    "IPK",
    "SKS_Lulus",
    "SKS_Diambil",
    "Sisa_Matkul_Wajib",
    "Tugas_Tertunda_Avg",
    "Skor_Psikologis_Avg",
    "Tidur_Avg_Jam",
    "Motivasi"
]

target = "Label"

X = df[fitur]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)

joblib.dump({
    "model": model,
    "fitur": fitur,
    "label": list(model.classes_)
}, "model.pkl")

with open("hasil_training.txt", "w", encoding="utf-8") as f:
    f.write("HASIL TRAINING MODEL CAKRA\n")
    f.write("===========================\n\n")
    f.write("Algoritma: Decision Tree Classifier\n")
    f.write("Max Depth : 4\n")
    f.write("Train-Test: 80% training dan 20% testing\n\n")

    f.write(f"Jumlah data training: {len(X_train)}\n")
    f.write(f"Jumlah data testing : {len(X_test)}\n")
    f.write(f"Akurasi model       : {akurasi * 100:.2f}%\n\n")

    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, zero_division=0))

    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)))

print("Training selesai!")
print(f"Akurasi model: {akurasi * 100:.2f}%")
print("Model berhasil disimpan sebagai model.pkl")
print("Hasil training disimpan sebagai hasil_training.txt")