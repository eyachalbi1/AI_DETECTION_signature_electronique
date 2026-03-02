#!/usr/bin/env python3
"""
Création d'un fichier CSV pour le dataset de signatures
"""

import os
import csv
import cv2
from config import *

def create_dataset_csv():
    """Créer un fichier CSV avec les informations du dataset"""
    
    csv_path = "dataset.csv"
    
    print("📄 Création du fichier CSV du dataset...")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # En-têtes
        writer.writerow(['filename', 'path', 'label', 'class', 'width', 'height', 'size_kb'])
        
        # Traiter les signatures authentiques
        if os.path.exists(GENUINE_DIR):
            for filename in os.listdir(GENUINE_DIR):
                if filename.endswith('.png'):
                    filepath = os.path.join(GENUINE_DIR, filename)
                    
                    # Obtenir les dimensions de l'image
                    img = cv2.imread(filepath)
                    if img is not None:
                        height, width = img.shape[:2]
                        size_kb = round(os.path.getsize(filepath) / 1024, 2)
                        
                        writer.writerow([
                            filename,
                            filepath,
                            1,  # Label numérique
                            'genuine',  # Label texte
                            width,
                            height,
                            size_kb
                        ])
        
        # Traiter les signatures falsifiées
        if os.path.exists(FORGED_DIR):
            for filename in os.listdir(FORGED_DIR):
                if filename.endswith('.png'):
                    filepath = os.path.join(FORGED_DIR, filename)
                    
                    # Obtenir les dimensions de l'image
                    img = cv2.imread(filepath)
                    if img is not None:
                        height, width = img.shape[:2]
                        size_kb = round(os.path.getsize(filepath) / 1024, 2)
                        
                        writer.writerow([
                            filename,
                            filepath,
                            0,  # Label numérique
                            'forged',  # Label texte
                            width,
                            height,
                            size_kb
                        ])
    
    # Statistiques
    genuine_count = len([f for f in os.listdir(GENUINE_DIR) if f.endswith('.png')]) if os.path.exists(GENUINE_DIR) else 0
    forged_count = len([f for f in os.listdir(FORGED_DIR) if f.endswith('.png')]) if os.path.exists(FORGED_DIR) else 0
    
    print(f"✅ Fichier CSV créé: {csv_path}")
    print(f"📊 Contenu:")
    print(f"   - {genuine_count} signatures authentiques")
    print(f"   - {forged_count} signatures falsifiées")
    print(f"   - {genuine_count + forged_count} images au total")

def create_results_csv():
    """Créer un fichier CSV pour stocker les résultats de prédiction"""
    
    csv_path = "results.csv"
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # En-têtes pour les résultats
        writer.writerow([
            'timestamp',
            'image_path', 
            'true_label',
            'predicted_label',
            'confidence',
            'correct'
        ])
    
    print(f"✅ Fichier de résultats créé: {csv_path}")
    print("💡 Utilisez ce fichier pour enregistrer les prédictions")

if __name__ == "__main__":
    create_dataset_csv()
    create_results_csv()