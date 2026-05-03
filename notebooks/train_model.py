import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# --- Chargement des données ---

df = pd.read_csv("data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")


# --- Encodage et features ---

le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = [
    'age', 'sexe_encoded', 'temperature', 'tension_sys',
    'toux', 'fatigue', 'maux_tete', 'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Cible : {y.shape}")


# --- Split train / test ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entraînement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")


# --- Entraînement ---

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Modèle entraîné : {type(model).__name__}")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Nombre de features : {model.n_features_in_}")
print(f"Classes : {list(model.classes_)}")


# --- Évaluation ---

y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prédiction': y_pred[:10]
})
print(comparison)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy:.2%}")

print("\nMatrice de confusion :")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))


# --- Visualisation ---

os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.xlabel('Prédiction du modèle')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.png', dpi=150)
plt.show()

print("Figure sauvegardée dans figures/confusion_matrix.png")


# --- Sauvegarde du modèle ---

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

size = os.path.getsize("models/model.pkl")
print(f"Modèle sauvegardé : models/model.pkl ({size / 1024:.1f} Ko)")
print("Encodeurs et metadata sauvegardés.")


# --- Test de rechargement ---

model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modèle rechargé : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")


# --- Prédiction sur un nouveau patient ---

nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'region': 'Dakar'
}

sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

features = [
    nouveau_patient['age'],
    sexe_enc,
    nouveau_patient['temperature'],
    nouveau_patient['tension_sys'],
    int(nouveau_patient['toux']),
    int(nouveau_patient['fatigue']),
    int(nouveau_patient['maux_tete']),
    region_enc
]

df_input = pd.DataFrame([features], columns=feature_cols)

diagnostic = model_loaded.predict(df_input)[0]
probas = model_loaded.predict_proba(df_input)[0]
proba_max = probas.max()

print("\n--- Résultat du pré-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")
print(f"Probabilité : {proba_max:.1%}")

print("\nProbabilités par classe :")
for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"  {classe:8s} : {proba:.1%} {bar}")


# --- Importance des features ---

print("\nImportance des features :")
for name, imp in sorted(zip(feature_cols, model.feature_importances_),
                        key=lambda x: x[1], reverse=True):
    print(f"  {name:15s} : {imp:.3f}")