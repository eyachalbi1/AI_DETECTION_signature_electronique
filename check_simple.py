import os

def check_files():
    print("=== VERIFICATION DU PROJET ===\n")
    
    # Fichiers essentiels
    essential = ["model.py", "train.py", "predict.py", "data_loader.py", "config.py", "requirements.txt"]
    
    print("FICHIERS ESSENTIELS:")
    missing = []
    for f in essential:
        if os.path.exists(f):
            print(f"  OK  {f}")
        else:
            print(f"  NOK {f}")
            missing.append(f)
    
    print("\nDOSSIERS:")
    dirs = ["dataset/genuine", "dataset/forged", "models", "logs"]
    for d in dirs:
        if os.path.exists(d):
            if "dataset" in d:
                count = len([x for x in os.listdir(d) if x.endswith('.png')])
                print(f"  OK  {d} ({count} images)")
            else:
                print(f"  OK  {d}")
        else:
            print(f"  NOK {d}")
            missing.append(d)
    
    print("\nMODELE:")
    model_path = "models/signature_detector.h5"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024*1024)
        print(f"  OK  {model_path} ({size:.1f} MB)")
    else:
        print(f"  NOK {model_path}")
        missing.append(model_path)
    
    print(f"\nRESUME:")
    if missing:
        print(f"  {len(missing)} elements manquants:")
        for m in missing:
            print(f"    - {m}")
        if model_path in missing:
            print("\n  ACTION: python train.py")
    else:
        print("  Projet complet!")

if __name__ == "__main__":
    check_files()