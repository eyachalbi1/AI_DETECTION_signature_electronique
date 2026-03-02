# Configuration du projet de détection de signatures

# Chemins des données
DATA_DIR = "dataset"
GENUINE_DIR = "dataset/genuine"
FORGED_DIR = "dataset/forged"
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