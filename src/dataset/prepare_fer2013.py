from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# Mapping des labels FER2013 -> noms de classes
EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def ensure_dir(path: Path):
    """Crée le dossier s'il n'existe pas."""
    path.mkdir(parents=True, exist_ok=True)


def prepare_fer2013():
    """
    Lit fer2013.csv et génère des images organisées par :
    data/processed/{train,val,test}/{emotion}/img_xxx.png
    """
    # On part du principe que ce script est dans src/dataset/
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    csv_path = raw_dir / "fer2013.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    print(f"Lecture du CSV : {csv_path}")
    df = pd.read_csv(csv_path)

    # FER2013 a normalement ces colonnes : emotion, pixels, Usage
    expected_cols = {"emotion", "pixels", "Usage"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Le CSV ne contient pas les colonnes attendues {expected_cols}. Colonnes trouvées : {df.columns}"
        )

    # On prépare les répertoires train/val/test
    for split in ["train", "val", "test"]:
        for label_name in EMOTION_LABELS.values():
            ensure_dir(processed_dir / split / label_name)

    # On mappe Usage -> split
    usage_to_split = {
        "Training": "train",
        "PublicTest": "val",     # souvent utilisé comme validation
        "PrivateTest": "test",   # souvent utilisé comme test
    }

    print("Génération des images...")
    counts = {"train": 0, "val": 0, "test": 0}

    for idx, row in df.iterrows():
        emotion_id = int(row["emotion"])
        pixels_str = row["pixels"]
        usage = row["Usage"]

        if usage not in usage_to_split:
            # Si cas bizarre, on ignore
            continue

        split = usage_to_split[usage]

        # Convertir la chaîne de pixels en tableau numpy
        pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")
        # FER2013 : images 48x48 en niveaux de gris
        if pixels.size != 48 * 48:
            # Cas anormal, on ignore
            continue

        img = pixels.reshape((48, 48))

        # Option 1 : garder en 48x48, niveaux de gris
        # Option 2 : redimensionner en 224x224 pour MobileNetV2 (qu'on fera plus tard)
        # Ici on garde 48x48 et on redimensionnera au moment du data loader.

        # Nom de la classe / dossier
        label_name = EMOTION_LABELS.get(emotion_id, None)
        if label_name is None:
            continue

        # Chemin de sortie
        out_dir = processed_dir / split / label_name
        # Exemple de nom : img_000123.png
        out_path = out_dir / f"img_{idx:06d}.png"

        # Sauvegarde de l'image (format PNG, niveaux de gris)
        cv2.imwrite(str(out_path), img)

        counts[split] += 1

        if idx % 1000 == 0:
            print(f"... {idx} lignes traitées")

    print("✅ Terminé.")
    print("Nombre d'images générées :")
    for split, c in counts.items():
        print(f"  {split}: {c}")


if __name__ == "__main__":
    prepare_fer2013()
