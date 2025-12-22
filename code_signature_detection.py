#!/usr/bin/env python3



import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================================
# CONFIGURATION
# ============================================================================

# Chemins des données
DATA_DIR = "dataset"
GENUINE_DIR = "dataset/genuine"
FORGED_DIR = "dataset/forged"
CSV_FILE = "dataset.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"

# Paramètres du modèle
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Paramètres d'entraînement
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Nom du modèle sauvegardé
MODEL_NAME = "signature_detector.h5"
MODEL_PATH = f"{MODEL_DIR}/{MODEL_NAME}"

# ============================================================================
# MODÈLE CNN
# ============================================================================

class SignatureDetector:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def preprocess_image(self, image_path):
        # Vérifier l'existence du fichier
        if not os.path.exists(image_path):
            print(f"Erreur: Fichier introuvable {image_path}")
            return None
            
        # Essayer différents encodages pour les chemins avec caractères spéciaux
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        except:
            try:
                img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            except:
                img = None
                
        if img is None:
            print(f"Erreur: Impossible de lire l'image {image_path}")
            print("Vérifiez le chemin et les caractères spéciaux")
            print("Formats supportés: .png, .jpg, .jpeg")
            return None
            
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=-1)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=EPOCHS):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=BATCH_SIZE
        )
    
    def predict(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        return 'genuine' if prediction[0][0] > 0.5 else 'forged'
    
    def analyze_signature(self, image_path):
        """Analyse complète avec texte descriptif"""
        image = self.preprocess_image(image_path)
        if image is None:
            return "Erreur: Impossible d'analyser l'image"
        
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        confidence = prediction[0][0]
        
        # Déterminer le résultat
        if confidence > 0.5:
            result = "AUTHENTIQUE"
            certainty = confidence
        else:
            result = "FALSIFIÉE"
            certainty = 1 - confidence
        
        # Niveau de confiance
        if certainty > 0.9:
            level = "très élevée"
        elif certainty > 0.7:
            level = "élevée"
        elif certainty > 0.6:
            level = "modérée"
        else:
            level = "faible"
        
        return f"Résultat: Cette signature est {result} avec une confiance {level} ({certainty:.1%})."
    
    def save_model(self, path):
        self.model.save(path)
    
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

# ============================================================================
# CHARGEUR DE DONNÉES
# ============================================================================

class DataLoader:
    def __init__(self):
        self.X = []
        self.y = []
    
    def load_data(self):
        """Charger les images du dataset"""
        print("Chargement des donnees...")
        
        # Vérifier si un fichier CSV existe
        if os.path.exists(CSV_FILE):
            print(f"Fichier CSV détecté: {CSV_FILE}")
            return self._load_from_csv(CSV_FILE)
        else:
            print("Chargement depuis les dossiers...")
            # Charger signatures authentiques (label = 1)
            genuine_count = self._load_from_directory(GENUINE_DIR, label=1)
            
            # Charger signatures falsifiées (label = 0)  
            forged_count = self._load_from_directory(FORGED_DIR, label=0)
            
            print(f"{genuine_count} signatures authentiques (genuine)")
            print(f"{forged_count} signatures falsifiees (forged)")
            
            return np.array(self.X), np.array(self.y)
    
    def _load_from_csv(self, csv_path):
        """Charger les données depuis un fichier CSV"""
        try:
            df = pd.read_csv(csv_path)
            print(f"CSV chargé: {len(df)} entrées")
            
            # Vérifier les colonnes requises
            required_cols = ['path', 'label']
            if not all(col in df.columns for col in required_cols):
                print(f"Erreur: Le CSV doit contenir les colonnes: {required_cols}")
                print(f"Colonnes trouvées: {list(df.columns)}")
                return np.array([]), np.array([])
            
            count = 0
            genuine_count = 0
            forged_count = 0
            
            for _, row in df.iterrows():
                img_path = row['path']
                label = row['label']
                
                # Convertir le label en format binaire
                if isinstance(label, str):
                    if label.lower() in ['genuine', 'authentic', '1', 'true']:
                        label = 1
                    else:
                        label = 0
                else:
                    label = int(label)
                
                # Charger l'image
                img = self._preprocess_image(img_path)
                if img is not None:
                    self.X.append(img)
                    self.y.append(label)
                    count += 1
                    if label == 1:
                        genuine_count += 1
                    else:
                        forged_count += 1
            
            print(f"{genuine_count} signatures authentiques")
            print(f"{forged_count} signatures falsifiées")
            print(f"Total: {count} images chargées")
            
            return np.array(self.X), np.array(self.y)
            
        except Exception as e:
            print(f"Erreur lors du chargement du CSV: {e}")
            return np.array([]), np.array([])
    
    def _load_from_directory(self, directory, label):
        """Charger images d'un dossier"""
        count = 0
        if not os.path.exists(directory):
            print(f"Dossier {directory} introuvable")
            return count
            
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(directory, filename)
                img = self._preprocess_image(img_path)
                if img is not None:
                    self.X.append(img)
                    self.y.append(label)
                    count += 1
        return count
    
    def _preprocess_image(self, image_path):
        """Préprocesser une image"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype('float32') / 255.0
            return np.expand_dims(img, axis=-1)
        except:
            print(f"Erreur lecture: {image_path}")
            return None
    
    def split_data(self, X, y):
        """Diviser les données"""
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=VALIDATION_SPLIT + TEST_SPLIT, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=TEST_SPLIT/(VALIDATION_SPLIT + TEST_SPLIT), random_state=42
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================================
# FONCTIONS D'ENTRAÎNEMENT
# ============================================================================

def train_model():
    print("Entrainement du modele de detection de signatures\n")
    
    # Charger les données
    loader = DataLoader()
    X, y = loader.load_data()
    
    if len(X) == 0:
        print("Aucune donnee trouvee.")
        print("Placez vos images dans dataset/genuine/ et dataset/forged/")
        return
    
    print(f"Total: {len(X)} images chargees\n")
    
    # Diviser les données
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    
    print(f"Division des donnees:")
    print(f"   Entrainement: {len(X_train)}")
    print(f"   Validation: {len(X_val)}")
    print(f"   Test: {len(X_test)}\n")
    
    # Créer le modèle
    detector = SignatureDetector(input_shape=(*IMG_SIZE, 1))
    
    print("Entrainement en cours...")
    history = detector.train(
        X_train, y_train, 
        X_val, y_val, 
        epochs=EPOCHS
    )
    
    # Évaluer sur les données de test
    test_loss, test_acc = detector.model.evaluate(X_test, y_test, verbose=0)
    print(f"\nPrecision sur test: {test_acc:.4f}")
    
    # Générer les graphiques d'entraînement
    plt.figure(figsize=(12, 4))
    
    # Graphique de précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entraînement', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation', color='red')
    plt.title('Precision du modele')
    plt.xlabel('Epoque')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graphique de perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entraînement', color='blue')
    plt.plot(history.history['val_loss'], label='Validation', color='red')
    plt.title('Perte du modele')
    plt.xlabel('Epoque')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    chart_path = 'training_history.png'
    plt.savefig(chart_path)
    plt.show()
    
    # Sauvegarder le modèle
    os.makedirs(MODEL_DIR, exist_ok=True)
    detector.save_model(MODEL_PATH)
    print(f"\nModele sauvegarde: {MODEL_PATH}")
    print(f"Graphique sauvegarde: {chart_path}")

def test_model(image_path):
    """Tester le modèle sur une image avec visualisation"""
    if not os.path.exists(MODEL_PATH):
        print("Modele non trouve. Entrainement requis.")
        return
    
    detector = SignatureDetector()
    detector.load_model(MODEL_PATH)
    
    # Analyser la signature
    result = detector.analyze_signature(image_path)
    print(result)
    
    # Afficher l'image avec le résultat
    img_original = cv2.imread(image_path)
    img_processed = detector.preprocess_image(image_path)
    
    if img_original is not None and img_processed is not None:
        plt.figure(figsize=(12, 5))
        
        # Image originale
        plt.subplot(1, 2, 1)
        if len(img_original.shape) == 3:
            plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img_original, cmap='gray')
        plt.title('Image originale')
        plt.axis('off')
        
        # Image préprocessée avec résultat
        plt.subplot(1, 2, 2)
        plt.imshow(img_processed.squeeze(), cmap='gray')
        plt.title(f'Image préprocessée\n{result}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def predict_random():
    """Prédire sur une image aléatoire du dataset"""
    if not os.path.exists(MODEL_PATH):
        print("Modele non trouve. Entrainement requis.")
        return
    
    # Collecter toutes les images
    images = []
    if os.path.exists(GENUINE_DIR):
        genuine_files = [os.path.join(GENUINE_DIR, f) for f in os.listdir(GENUINE_DIR) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images.extend(genuine_files)
    
    if os.path.exists(FORGED_DIR):
        forged_files = [os.path.join(FORGED_DIR, f) for f in os.listdir(FORGED_DIR) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images.extend(forged_files)
    
    if not images:
        print("Aucune image trouvée dans le dataset")
        return
    
    # Sélectionner une image aléatoire
    selected_image = random.choice(images)
    true_label = "genuine" if "genuine" in selected_image else "forged"
    
    print(f"Image sélectionnée: {os.path.basename(selected_image)}")
    print(f"Type réel: {true_label.upper()}")
    print("-" * 50)
    
    # Tester l'image avec visualisation
    test_model(selected_image)

def evaluate_model():
    """Evaluer les performances du modèle"""
    if not os.path.exists(MODEL_PATH):
        print("Modele non trouve. Entrainement requis.")
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
    
    # Evaluer sur les données de test
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

def main():
    """Fonction principale"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python code_signature_detection.py train")
        print("  python code_signature_detection.py test [image_path|random]")
        print("  python code_signature_detection.py <image_path>")
        print("  python code_signature_detection.py evaluate")
        return
    
    command = sys.argv[1]
    
    if command == "train":
        train_model()
    elif command == "test":
        if len(sys.argv) == 3:
            if sys.argv[2].lower() == "random":
                predict_random()  # Test avec image aléatoire
            else:
                test_model(sys.argv[2])  # Test avec image spécifiée
        else:
            predict_random()  # Par défaut: image aléatoire si pas d'argument
    elif command == "evaluate":
        evaluate_model()
    elif os.path.exists(command):  # Si c'est un chemin d'image valide
        test_model(command)
    else:
        print("Commande invalide ou fichier introuvable")
        print("Utilisez 'train', 'test [image|random]', 'evaluate' ou directement <image_path>")

if __name__ == "__main__":
    main()