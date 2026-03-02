#!/usr/bin/env python3
"""
Prédiction en lot avec sauvegarde CSV
"""

import os
import csv
import pandas as pd
from datetime import datetime
from model import SignatureDetector
from config import *

def predict_batch_to_csv():
    """Faire des prédictions en lot et sauvegarder dans CSV"""
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Modèle non trouvé. Entraînez d'abord le modèle.")
        return
    
    # Charger le modèle
    detector = SignatureDetector(input_shape=(*IMG_SIZE, 1))
    detector.load_model(MODEL_PATH)
    
    results = []
    
    print("🔄 Prédictions en cours...")
    
    # Prédire sur quelques images de test
    test_images = []
    
    # Prendre 10 images de chaque classe
    if os.path.exists(GENUINE_DIR):
        genuine_files = [f for f in os.listdir(GENUINE_DIR) if f.endswith('.png')][:10]
        for filename in genuine_files:
            test_images.append((os.path.join(GENUINE_DIR, filename), 'genuine'))
    
    if os.path.exists(FORGED_DIR):
        forged_files = [f for f in os.listdir(FORGED_DIR) if f.endswith('.png')][:10]
        for filename in forged_files:
            test_images.append((os.path.join(FORGED_DIR, filename), 'forged'))
    
    # Faire les prédictions
    for img_path, true_label in test_images:
        try:
            image = detector.preprocess_image(img_path)
            predicted_label = detector.predict(image)
            confidence = detector.model.predict(image.reshape(1, *image.shape))[0][0]
            
            # Si prédiction est 'genuine', confidence est directe
            # Si prédiction est 'forged', confidence = 1 - confidence
            if predicted_label == 'forged':
                confidence = 1 - confidence
            
            correct = (predicted_label == true_label)
            
            results.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'image_path': img_path,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': round(confidence, 4),
                'correct': correct
            })
            
            status = "✅" if correct else "❌"
            print(f"{status} {os.path.basename(img_path)}: {predicted_label} (conf: {confidence:.3f})")
            
        except Exception as e:
            print(f"❌ Erreur avec {img_path}: {e}")
    
    # Sauvegarder dans CSV
    csv_path = "prediction_results.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        if results:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    # Statistiques
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / len(results)
        
        print(f"\n📊 Résultats sauvegardés dans: {csv_path}")
        print(f"   Total testé: {len(results)}")
        print(f"   Correct: {correct_count}")
        print(f"   Précision: {accuracy:.2%}")
    
    return csv_path

def analyze_csv_results():
    """Analyser les résultats CSV"""
    
    csv_path = "prediction_results.csv"
    
    if not os.path.exists(csv_path):
        print("❌ Fichier de résultats non trouvé. Lancez d'abord predict_batch_to_csv()")
        return
    
    # Lire le CSV
    df = pd.read_csv(csv_path)
    
    print("📈 Analyse des résultats:")
    print(f"   Total d'images: {len(df)}")
    print(f"   Précision globale: {df['correct'].mean():.2%}")
    
    # Par classe
    for label in ['genuine', 'forged']:
        subset = df[df['true_label'] == label]
        if len(subset) > 0:
            accuracy = subset['correct'].mean()
            avg_confidence = subset['confidence'].mean()
            print(f"   {label.capitalize()}: {accuracy:.2%} (conf moy: {avg_confidence:.3f})")

if __name__ == "__main__":
    csv_file = predict_batch_to_csv()
    print()
    analyze_csv_results()