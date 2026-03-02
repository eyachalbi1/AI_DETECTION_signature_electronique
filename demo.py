#!/usr/bin/env python3
"""
Démonstration du système de détection de signatures
"""

import os
import random
from model import SignatureDetector
from config import *

def run_demo():
    """Démonstration avec des images aléatoires du dataset"""
    
    print("🎬 Démonstration du système de détection de signatures\n")
    
    # Vérifier le modèle
    if not os.path.exists(MODEL_PATH):
        print("❌ Modèle non trouvé. Entraînez d'abord le modèle avec: python train.py")
        return
    
    # Charger le modèle
    detector = SignatureDetector(input_shape=(*IMG_SIZE, 1))
    detector.load_model(MODEL_PATH)
    print("✅ Modèle chargé\n")
    
    # Sélectionner des images aléatoirement
    genuine_images = [f for f in os.listdir(GENUINE_DIR) if f.endswith('.png')][:5]
    forged_images = [f for f in os.listdir(FORGED_DIR) if f.endswith('.png')][:5]
    
    print("🔍 Test sur signatures authentiques:")
    print("-" * 50)
    
    correct_genuine = 0
    for img_name in genuine_images:
        img_path = os.path.join(GENUINE_DIR, img_name)
        image = detector.preprocess_image(img_path)
        prediction = detector.predict(image)
        confidence = detector.model.predict(image.reshape(1, *image.shape))[0][0]
        
        status = "✅" if prediction == "genuine" else "❌"
        if prediction == "genuine":
            correct_genuine += 1
            
        print(f"{status} {img_name}: {prediction} (confiance: {confidence:.3f})")
    
    print(f"\n🔍 Test sur signatures falsifiées:")
    print("-" * 50)
    
    correct_forged = 0
    for img_name in forged_images:
        img_path = os.path.join(FORGED_DIR, img_name)
        image = detector.preprocess_image(img_path)
        prediction = detector.predict(image)
        confidence = detector.model.predict(image.reshape(1, *image.shape))[0][0]
        
        status = "✅" if prediction == "forged" else "❌"
        if prediction == "forged":
            correct_forged += 1
            
        print(f"{status} {img_name}: {prediction} (confiance: {confidence:.3f})")
    
    # Statistiques
    total_correct = correct_genuine + correct_forged
    total_tested = len(genuine_images) + len(forged_images)
    accuracy = total_correct / total_tested
    
    print(f"\n📊 Résultats de la démonstration:")
    print(f"   Authentiques correctes: {correct_genuine}/{len(genuine_images)}")
    print(f"   Falsifiées correctes: {correct_forged}/{len(forged_images)}")
    print(f"   Précision globale: {accuracy:.2%}")
    
    print(f"\n💡 Pour tester vos propres images:")
    print(f"   python predict.py chemin/vers/votre/image.png")
    print(f"   python gui.py  # Interface graphique")

if __name__ == "__main__":
    run_demo()