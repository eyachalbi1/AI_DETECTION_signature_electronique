#!/usr/bin/env python3
"""
Vérification complète des fichiers du projet
"""

import os

def check_project_files():
    """Vérifier tous les fichiers nécessaires"""
    
    print("Verification des fichiers du projet\n")
    
    # Fichiers essentiels
    essential_files = {
        "model.py": "Architecture du CNN",
        "train.py": "Script d'entraînement", 
        "predict.py": "Script de prédiction",
        "data_loader.py": "Chargement des données",
        "config.py": "Configuration du projet",
        "requirements.txt": "Dépendances Python",
        "README.md": "Documentation"
    }
    
    # Fichiers utilitaires
    utility_files = {
        "evaluate.py": "Évaluation du modèle",
        "gui.py": "Interface graphique",
        "demo.py": "Démonstration",
        "create_csv.py": "Génération CSV",
        "predict_batch.py": "Prédictions en lot",
        "simple_predict.py": "Prédiction simple"
    }
    
    # Dossiers nécessaires
    required_dirs = {
        "dataset/": "Dossier des données",
        "dataset/genuine/": "Signatures authentiques",
        "dataset/forged/": "Signatures falsifiées",
        "models/": "Modèles sauvegardés",
        "logs/": "Logs d'entraînement",
        "signature_env/": "Environnement Python"
    }
    
    # Fichiers de configuration
    config_files = {
        ".vscode/settings.json": "Configuration VSCode (optionnel)",
        "dataset.csv": "Index du dataset (généré)",
        "results.csv": "Résultats (généré)"
    }
    
    missing_files = []
    
    print("📋 FICHIERS ESSENTIELS:")
    print("-" * 40)
    for file, desc in essential_files.items():
        if os.path.exists(file):
            print(f"✅ {file:<20} - {desc}")
        else:
            print(f"❌ {file:<20} - {desc}")
            missing_files.append(file)
    
    print(f"\n🛠️ FICHIERS UTILITAIRES:")
    print("-" * 40)
    for file, desc in utility_files.items():
        if os.path.exists(file):
            print(f"✅ {file:<20} - {desc}")
        else:
            print(f"⚠️ {file:<20} - {desc}")
    
    print(f"\n📁 DOSSIERS:")
    print("-" * 40)
    for dir_path, desc in required_dirs.items():
        if os.path.exists(dir_path):
            if dir_path.startswith("dataset/"):
                # Compter les images
                if dir_path.endswith("genuine/"):
                    count = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
                    print(f"✅ {dir_path:<20} - {desc} ({count} images)")
                elif dir_path.endswith("forged/"):
                    count = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
                    print(f"✅ {dir_path:<20} - {desc} ({count} images)")
                else:
                    print(f"✅ {dir_path:<20} - {desc}")
            else:
                print(f"✅ {dir_path:<20} - {desc}")
        else:
            print(f"❌ {dir_path:<20} - {desc}")
            missing_files.append(dir_path)
    
    print(f"\n⚙️ FICHIERS DE CONFIGURATION:")
    print("-" * 40)
    for file, desc in config_files.items():
        if os.path.exists(file):
            print(f"✅ {file:<20} - {desc}")
        else:
            print(f"⚠️ {file:<20} - {desc}")
    
    # Vérifier le modèle entraîné
    print(f"\n🧠 MODÈLE ENTRAÎNÉ:")
    print("-" * 40)
    model_path = "models/signature_detector.h5"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"✅ {model_path:<20} - Modèle CNN ({size_mb:.1f} MB)")
    else:
        print(f"❌ {model_path:<20} - Modèle CNN (à entraîner)")
        missing_files.append(model_path)
    
    # Résumé
    print(f"\n📊 RÉSUMÉ:")
    print("-" * 40)
    if missing_files:
        print(f"❌ {len(missing_files)} fichiers/dossiers manquants:")
        for item in missing_files:
            print(f"   - {item}")
        
        print(f"\n💡 ACTIONS À FAIRE:")
        if "models/signature_detector.h5" in missing_files:
            print("   1. Entraîner le modèle: python train.py")
        if any("dataset/" in item for item in missing_files):
            print("   2. Créer les dossiers: mkdir dataset\\genuine dataset\\forged")
        if ".vscode/settings.json" in missing_files:
            print("   3. Configurer VSCode pour l'interpréteur Python")
    else:
        print("✅ Tous les fichiers essentiels sont présents!")
        print("🚀 Projet prêt à utiliser!")
    
    return len(missing_files) == 0

if __name__ == "__main__":
    check_project_files()