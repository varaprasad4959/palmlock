import os
import cv2
import numpy as np

from recognition import PalmRecognizer


ENROLLMENT_DIR = r"C:\Users\varap\PalmLock\data\enrollment"

TEMPLATE_PATH = (
    r"C:\Users\varap\PalmLock\data\enrollment\palm_template.npy"
)

# Temporary database is not needed.
# We use the recognizer's feature extractor only.
DUMMY_DATABASE = os.path.join(
    ENROLLMENT_DIR,
    "_empty_database.npy"
)

# Create a temporary empty database so PalmRecognizer can initialize.
np.save(
    DUMMY_DATABASE,
    np.empty((0, 512), dtype=np.float32)
)

recognizer = PalmRecognizer(DUMMY_DATABASE)

embeddings = []

files = sorted(
    f for f in os.listdir(ENROLLMENT_DIR)
    if f.lower().endswith(".jpg")
)

print(f"Found {len(files)} enrollment images.")

for i, filename in enumerate(files):

    path = os.path.join(
        ENROLLMENT_DIR,
        filename
    )

    image = cv2.imread(path)

    if image is None:
        print(f"Skipping {filename}")
        continue

    embedding = recognizer.extract_embedding(image)

    if embedding is not None:
        embeddings.append(embedding)

    print(
        f"Processed {i + 1}/{len(files)}"
    )


if not embeddings:
    raise RuntimeError(
        "No palm embeddings could be created."
    )


embeddings = np.asarray(
    embeddings,
    dtype=np.float32
)

# Normalize again for safety.
norms = np.linalg.norm(
    embeddings,
    axis=1,
    keepdims=True
)

embeddings = embeddings / np.maximum(
    norms,
    1e-12
)

np.save(
    TEMPLATE_PATH,
    embeddings
)

# Remove temporary database.
os.remove(DUMMY_DATABASE)

print()
print("==============================")
print("PALM TEMPLATE CREATED")
print("==============================")
print("Samples:", len(embeddings))
print("Embedding shape:", embeddings.shape)
print("Saved:", TEMPLATE_PATH)
print("==============================")