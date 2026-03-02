import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from config import *

class DataLoader:
    def __init__(self):
        self.X = []
        self.y = []
    
    def load_data(self):
        """Charger les images du dataset"""
        print("Chargement des donnees...")
        
        # Charger signatures authentiques (label = 1)
        genuine_count = self._load_from_directory(GENUINE_DIR, label=1)
        
        # Charger signatures falsifiées (label = 0)  
        forged_count = self._load_from_directory(FORGED_DIR, label=0)
        
        print(f"{genuine_count} signatures authentiques (genuine)")
        print(f"{forged_count} signatures falsifiees (forged)")
        
        return np.array(self.X), np.array(self.y)
    
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