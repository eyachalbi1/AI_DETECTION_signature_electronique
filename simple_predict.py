#!/usr/bin/env python3
"""
Prédiction simple avec texte descriptif
"""

import sys
from model import SignatureDetector
from config import *

def simple_prediction(image_path):
    """Prédiction avec texte simple"""
    
    if not os.path.exists(MODEL_PATH):
        return "❌ Modèle non trouvé. Entraînez d'abord le modèle."
    
    if not os.path.exists(image_path):
        return f"❌ Image non trouvée: {image_path}"
    
    # Charger le modèle
    detector = SignatureDetector(input_shape=(*IMG_SIZE, 1))
    detector.load_model(MODEL_PATH)
    
    # Analyse complète
    result = detector.analyze_signature(image_path)
    
    return result

def main():
    """Interface en ligne de commande"""
    
    if len(sys.argv) != 2:
        print("Usage: python simple_predict.py <chemin_image>")
        print("Exemple: python simple_predict.py dataset/genuine/genuine_0001.png")
        return
    
    image_path = sys.argv[1]
    result = simple_prediction(image_path)
    
    print(f"🔍 Analyse de: {image_path}")
    print(f"📋 {result}")

if __name__ == "__main__":
    main()