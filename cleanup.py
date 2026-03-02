#!/usr/bin/env python3
"""
Script de nettoyage automatique du projet
"""

import os
import shutil

def cleanup_project():
    """Supprimer les fichiers non nécessaires"""
    
    files_to_remove = [
        # Fichiers temporaires
        "dataset/archive",
        "signature_env_old",
        
        # Fichiers .gitkeep
        "dataset/genuine/.gitkeep",
        "dataset/forged/.gitkeep",
        
        # Scripts utilitaires optionnels
        "organize_dataset.py",
        "setup_env.py", 
        "test_env.py",
        "start.py",
        "utils.py"
    ]
    
    removed_count = 0
    space_saved = 0
    
    print("🧹 Nettoyage du projet en cours...\n")
    
    for item in files_to_remove:
        if os.path.exists(item):
            try:
                # Calculer la taille avant suppression
                if os.path.isdir(item):
                    size = get_dir_size(item)
                    shutil.rmtree(item)
                    print(f"📁 Dossier supprimé: {item} ({format_size(size)})")
                else:
                    size = os.path.getsize(item)
                    os.remove(item)
                    print(f"📄 Fichier supprimé: {item} ({format_size(size)})")
                
                space_saved += size
                removed_count += 1
                
            except Exception as e:
                print(f"❌ Erreur avec {item}: {e}")
        else:
            print(f"⚠️ Introuvable: {item}")
    
    print(f"\n✅ Nettoyage terminé!")
    print(f"📊 {removed_count} éléments supprimés")
    print(f"💾 Espace libéré: {format_size(space_saved)}")
    
    # Afficher la structure finale
    print(f"\n📁 Structure finale du projet:")
    show_final_structure()

def get_dir_size(path):
    """Calculer la taille d'un dossier"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except:
        pass
    return total

def format_size(bytes):
    """Formater la taille en unités lisibles"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} TB"

def show_final_structure():
    """Afficher la structure finale"""
    essential_files = [
        "model.py",
        "train.py", 
        "predict.py",
        "data_loader.py",
        "config.py",
        "requirements.txt",
        "README.md"
    ]
    
    for file in essential_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (manquant)")
    
    # Compter les images
    genuine_count = len([f for f in os.listdir("dataset/genuine") if f.endswith('.png')]) if os.path.exists("dataset/genuine") else 0
    forged_count = len([f for f in os.listdir("dataset/forged") if f.endswith('.png')]) if os.path.exists("dataset/forged") else 0
    
    print(f"   📊 Dataset: {genuine_count} genuine + {forged_count} forged = {genuine_count + forged_count} images")

if __name__ == "__main__":
    cleanup_project()