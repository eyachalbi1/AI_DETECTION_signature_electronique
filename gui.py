#!/usr/bin/env python3
"""
Interface graphique simple pour la détection de signatures
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from model import SignatureDetector
from config import *

class SignatureGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Détection de Signatures")
        self.root.geometry("600x500")
        
        self.detector = None
        self.load_model()
        
        self.setup_ui()
    
    def load_model(self):
        """Charger le modèle"""
        try:
            if os.path.exists(MODEL_PATH):
                self.detector = SignatureDetector(input_shape=(*IMG_SIZE, 1))
                self.detector.load_model(MODEL_PATH)
                print("✅ Modèle chargé")
            else:
                messagebox.showerror("Erreur", "Modèle non trouvé. Entraînez d'abord le modèle.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de chargement: {e}")
    
    def setup_ui(self):
        """Configurer l'interface"""
        
        # Titre
        title = tk.Label(self.root, text="🔍 Détection de Signatures", 
                        font=("Arial", 20, "bold"))
        title.pack(pady=20)
        
        # Bouton de sélection d'image
        self.select_btn = tk.Button(self.root, text="📁 Sélectionner une image", 
                                   command=self.select_image, 
                                   font=("Arial", 12),
                                   bg="#4CAF50", fg="white",
                                   width=20, height=2)
        self.select_btn.pack(pady=10)
        
        # Zone d'affichage de l'image
        self.image_frame = tk.Frame(self.root, bg="lightgray", width=300, height=200)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame, text="Aucune image sélectionnée", 
                                   bg="lightgray")
        self.image_label.pack(expand=True)
        
        # Bouton de prédiction
        self.predict_btn = tk.Button(self.root, text="🎯 Analyser", 
                                    command=self.predict_signature,
                                    font=("Arial", 12),
                                    bg="#2196F3", fg="white",
                                    width=20, height=2,
                                    state="disabled")
        self.predict_btn.pack(pady=10)
        
        # Zone de résultat
        self.result_frame = tk.Frame(self.root)
        self.result_frame.pack(pady=20)
        
        self.result_label = tk.Label(self.result_frame, text="", 
                                    font=("Arial", 16, "bold"))
        self.result_label.pack()
        
        self.confidence_label = tk.Label(self.result_frame, text="", 
                                        font=("Arial", 12))
        self.confidence_label.pack()
    
    def select_image(self):
        """Sélectionner une image"""
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image de signature",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_image(file_path)
            self.predict_btn.config(state="normal")
            self.result_label.config(text="")
            self.confidence_label.config(text="")
    
    def display_image(self, path):
        """Afficher l'image sélectionnée"""
        try:
            # Charger et redimensionner l'image
            image = Image.open(path)
            image.thumbnail((250, 150), Image.Resampling.LANCZOS)
            
            # Convertir pour tkinter
            photo = ImageTk.PhotoImage(image)
            
            # Afficher
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Garder une référence
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher l'image: {e}")
    
    def predict_signature(self):
        """Faire une prédiction"""
        if not self.detector:
            messagebox.showerror("Erreur", "Modèle non chargé")
            return
        
        try:
            # Préprocesser l'image
            image = self.detector.preprocess_image(self.image_path)
            
            # Prédiction
            result = self.detector.predict(image)
            confidence = self.detector.model.predict(np.expand_dims(image, axis=0))[0][0]
            
            # Afficher le résultat
            if result == "genuine":
                self.result_label.config(text="✅ SIGNATURE AUTHENTIQUE", fg="green")
            else:
                self.result_label.config(text="❌ SIGNATURE FALSIFIÉE", fg="red")
            
            self.confidence_label.config(text=f"Confiance: {confidence:.2%}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur de prédiction: {e}")
    
    def run(self):
        """Lancer l'interface"""
        self.root.mainloop()

if __name__ == "__main__":
    app = SignatureGUI()
    app.run()