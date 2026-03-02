#!/usr/bin/env python3
"""
Évaluation du modèle de détection de signatures
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from model import SignatureDetector
from data_loader import DataLoader
from config import *

def evaluate_model():
    """Évaluer les performances du modèle"""
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Modèle non trouvé. Entraînez d'abord le modèle avec: python train.py")
        return
    
    print("Evaluation du modele...\n")
    
    # Charger les données
    loader = DataLoader()
    X, y = loader.load_data()
    
    if len(X) == 0:
        print("Aucune donnee trouvee")
        return
    
    # Diviser les données
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    
    # Charger le modèle
    detector = SignatureDetector(input_shape=(*IMG_SIZE, 1))
    detector.load_model(MODEL_PATH)
    
    # Évaluer sur les données de test
    test_loss, test_acc = detector.model.evaluate(X_test, y_test, verbose=0)
    
    # Prédictions
    y_pred = detector.model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    
    # Afficher les résultats
    print(f"Precision sur test: {test_acc:.4f}")
    print(f"Perte sur test: {test_loss:.4f}\n")
    
    # Rapport de classification
    print("Rapport de classification:")
    print(classification_report(y_test, y_pred_binary, 
                              target_names=['Forged', 'Genuine']))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_binary)
    print(f"\nMatrice de confusion:")
    print(f"              Prédiction")
    print(f"Réalité    Forged  Genuine")
    print(f"Forged       {cm[0,0]:3d}     {cm[0,1]:3d}")
    print(f"Genuine      {cm[1,0]:3d}     {cm[1,1]:3d}")
    
    # Calculer les métriques
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetriques detaillees:")
    print(f"   Précision: {precision:.4f}")
    print(f"   Rappel: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

if __name__ == "__main__":
    evaluate_model()