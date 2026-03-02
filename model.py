import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np

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
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        img = img.astype('float32') / 255.0
        return np.expand_dims(img, axis=-1)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20):
        return self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32
        )
    
    def predict(self, image):
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        return 'genuine' if prediction[0][0] > 0.5 else 'forged'
    
    def predict_with_text(self, image):
        """Prédiction avec texte descriptif"""
        prediction = self.model.predict(np.expand_dims(image, axis=0))
        confidence = prediction[0][0]
        
        if confidence > 0.8:
            return "Cette signature est AUTHENTIQUE avec une forte confiance."
        elif confidence > 0.5:
            return "Cette signature semble AUTHENTIQUE mais avec une confiance modérée."
        elif confidence > 0.2:
            return "Cette signature semble FALSIFIÉE mais avec une confiance modérée."
        else:
            return "Cette signature est FALSIFIÉE avec une forte confiance."
    
    def analyze_signature(self, image_path):
        """Analyse complète avec texte descriptif"""
        image = self.preprocess_image(image_path)
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