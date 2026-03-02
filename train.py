from model import SignatureDetector
from data_loader import DataLoader
from config import *
import os

def main():
    print("🚀 Entraînement du modèle de détection de signatures\n")
    
    # Charger les données
    loader = DataLoader()
    X, y = loader.load_data()
    
    if len(X) == 0:
        print("❌ Aucune donnée trouvée.")
        print("📁 Placez vos images dans dataset/genuine/ et dataset/forged/")
        return
    
    print(f"📊 Total: {len(X)} images chargées\n")
    
    # Diviser les données
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    
    print(f"🔄 Division des données:")
    print(f"   Entraînement: {len(X_train)}")
    print(f"   Validation: {len(X_val)}")
    print(f"   Test: {len(X_test)}\n")
    
    # Créer le modèle
    detector = SignatureDetector(input_shape=(*IMG_SIZE, 1))
    
    print("🎯 Entraînement en cours...")
    history = detector.train(
        X_train, y_train, 
        X_val, y_val, 
        epochs=EPOCHS
    )
    
    # Évaluer sur les données de test
    test_loss, test_acc = detector.model.evaluate(X_test, y_test, verbose=0)
    print(f"\nPrécision sur test: {test_acc:.4f}")
    
    # Générer les graphiques d'entraînement
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    # Graphique de précision
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entraînement', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation', color='red')
    plt.title('Précision du modèle')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Graphique de perte
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entraînement', color='blue')
    plt.plot(history.history['val_loss'], label='Validation', color='red')
    plt.title('Perte du modèle')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    chart_path = 'training_history.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Graphiques sauvegardés: {chart_path}")
    
    # Afficher le graphique
    plt.show()
    
    # Sauvegarder le modèle
    os.makedirs(MODEL_DIR, exist_ok=True)
    detector.save_model(MODEL_PATH)
    print(f"Modèle sauvegardé: {MODEL_PATH}")

if __name__ == "__main__":
    main()