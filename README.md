# Détection de Signatures

Projet de détection de signatures authentiques vs falsifiées utilisant un CNN.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Préparer les données
Placez vos images dans:
- `dataset/real/` : signatures authentiques
- `dataset/fake/` : signatures falsifiées

### 2. Entraîner le modèle
```bash
python train.py
```

### 3. Faire des prédictions
```bash
python predict.py chemin/vers/image.png
```

## Structure
- `model.py` : Architecture du CNN
- `train.py` : Script d'entraînement
- `predict.py` : Script de prédiction
- `dataset/` : Dossier des données d'entraînement