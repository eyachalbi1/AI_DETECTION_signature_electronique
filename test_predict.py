#!/usr/bin/env python3
"""
Test simple de prédiction sans conflit
"""

import os
import sys

def test_imports():
    """Tester les imports un par un"""
    try:
        print("Test import cv2...")
        import cv2
        print("OK - cv2")
        
        print("Test import numpy...")
        import numpy as np
        print("OK - numpy")
        
        print("Test import tensorflow...")
        import tensorflow as tf
        print("OK - tensorflow")
        
        print("Test import keras...")
        from tensorflow import keras
        print("OK - keras")
        
        return True
    except Exception as e:
        print(f"Erreur: {e}")
        return False

def simple_predict():
    """Prédiction simple sans les imports problématiques"""
    
    # Vérifier si le modèle existe
    model_path = "models/signature_detector.h5"
    if not os.path.exists(model_path):
        print("Modèle non trouvé. Entraînez d'abord avec: python train.py")
        return
    
    print("Modèle trouvé!")
    
    # Vérifier les images
    genuine_dir = "dataset/genuine"
    forged_dir = "dataset/forged"
    
    if os.path.exists(genuine_dir):
        genuine_count = len([f for f in os.listdir(genuine_dir) if f.endswith('.png')])
        print(f"Images genuine: {genuine_count}")
    
    if os.path.exists(forged_dir):
        forged_count = len([f for f in os.listdir(forged_dir) if f.endswith('.png')])
        print(f"Images forged: {forged_count}")
    
    print("Dataset OK!")

if __name__ == "__main__":
    print("=== TEST ENVIRONNEMENT ===")
    
    if test_imports():
        print("\n=== TEST PREDICTION ===")
        simple_predict()
    else:
        print("Problème avec les imports")