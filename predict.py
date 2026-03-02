import sys
import os
from model import SignatureDetector
import cv2
import matplotlib.pyplot as plt

def predict_signature(image_path, model_path='models/signature_detector.h5'):
    if not os.path.exists(model_path):
        print("Erreur: Modèle non trouvé. Entraînez d'abord le modèle avec train.py")
        return
    
    if not os.path.exists(image_path):
        print(f"Erreur: Image non trouvée: {image_path}")
        return
    
    # Charger le modèle
    detector = SignatureDetector()
    detector.load_model(model_path)
    
    # Préprocesser l'image
    image = detector.preprocess_image(image_path)
    
    # Faire la prédiction
    result = detector.predict(image)
    confidence = detector.model.predict(image.reshape(1, 128, 128, 1))[0][0]
    
    # Afficher les résultats
    original_img = cv2.imread(image_path)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Image originale')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Prédiction: {result}\nConfiance: {confidence:.2f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Résultat: {result}")
    print(f"Confiance: {confidence:.2f}")
    
    # Analyse avec texte descriptif
    text_result = detector.analyze_signature(image_path)
    print(f"Analyse: {text_result}")
    
    # Vérifier si c'est correct (pour les images du dataset)
    if "genuine" in image_path or "forged" in image_path:
        true_label = "genuine" if "genuine" in image_path else "forged"
        is_correct = (result == true_label)
        status = "CORRECT" if is_correct else "INCORRECT"
        print(f"Verification: {status} (attendu: {true_label.upper()})")
    
    return result, confidence

def predict_random_from_dataset():
    """Prédire sur des images aléatoires du dataset"""
    import random
    from config import GENUINE_DIR, FORGED_DIR
    
    # Collecter les images disponibles
    images = []
    
    if os.path.exists(GENUINE_DIR):
        genuine_files = [os.path.join(GENUINE_DIR, f) for f in os.listdir(GENUINE_DIR) if f.endswith('.png')]
        images.extend(genuine_files)  # Toutes les images genuine
    
    if os.path.exists(FORGED_DIR):
        forged_files = [os.path.join(FORGED_DIR, f) for f in os.listdir(FORGED_DIR) if f.endswith('.png')]
        images.extend(forged_files)  # Toutes les images forged
    
    if not images:
        print("❌ Aucune image trouvée dans le dataset")
        return
    
    # Sélectionner une image aléatoire
    selected_image = random.choice(images)
    
    # Déterminer le type réel
    true_label = "genuine" if "genuine" in selected_image else "forged"
    
    print(f"Image selectionnee: {os.path.basename(selected_image)}")
    print(f"Type reel: {true_label.upper()}")
    print(f"Chemin: {selected_image}")
    print("-" * 60)
    
    predict_signature(selected_image)

def main():
    if len(sys.argv) == 1:
        # Aucun argument = prédiction aléatoire
        predict_random_from_dataset()
    elif len(sys.argv) == 2:
        # Un argument = image spécifique
        image_path = sys.argv[1]
        predict_signature(image_path)
    else:
        print("Usage:")
        print("  python predict.py                    # Image aléatoire du dataset")
        print("  python predict.py <chemin_image>     # Image spécifique")
        return

if __name__ == "__main__":
    main()