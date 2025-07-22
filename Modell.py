from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from Datenaufbereitung import X_train, X_test, y_train, y_test

# Modell initialisieren
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

# Training auf den TF-IDF-Vektoren
model.fit(X_train, y_train)

# Vorhersagen auf den Testdaten
y_pred = model.predict(X_test)

if __name__ == '__main__':
    # Accuracy – wie viele korrekt?
    acc = accuracy_score(y_test, y_pred)
    print(f"Genauigkeit (Accuracy): {acc:.4f}")

    # Confusion Matrix – welche Klassen verwechselt das Modell?
    print("Konfusionsmatrix:")
    print(confusion_matrix(y_test, y_pred))

    # Detaillierter Report: Precision, Recall, F1-Score je Klasse
    print("Klassifikationsbericht:")
    print(classification_report(y_test, y_pred))
