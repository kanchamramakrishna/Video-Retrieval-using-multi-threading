import os
import cv2
import numpy as np
import pandas as pd

train_folder = r"C:\Users\JAIVIVEKNAGABHYRU\Documents\vscode\dataset\train"
feature_folder = r"C:\Users\JAIVIVEKNAGABHYRU\Documents\vscode\dataset\features"

os.makedirs(feature_folder, exist_ok=True)

metadata = []

for filename in os.listdir(train_folder):
    if not filename.lower().endswith(".avi"):
        continue

    video_path = os.path.join(train_folder, filename)
    feature_path = os.path.join(feature_folder, filename.replace(".avi", ".npy"))

    cap = cv2.VideoCapture(video_path)

    features = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        r_hist = cv2.calcHist([frame], [0], None, [8], [0, 256])
        g_hist = cv2.calcHist([frame], [1], None, [8], [0, 256])
        b_hist = cv2.calcHist([frame], [2], None, [8], [0, 256])

        r_hist = cv2.normalize(r_hist, r_hist).flatten()
        g_hist = cv2.normalize(g_hist, g_hist).flatten()
        b_hist = cv2.normalize(b_hist, b_hist).flatten()

        feature = np.concatenate([r_hist, g_hist, b_hist])

        features.append(feature)

    cap.release()

    features = np.array(features)

    if len(features) == 0:
        print("Skipped:", filename)
        continue

    np.save(feature_path, features)

    metadata.append({
        "video": filename,
        "feature_file": os.path.basename(feature_path),
        "frames": len(features),
        "feature_dimension": features.shape[1]
    })

    print(filename, "->", features.shape)

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(
    os.path.join(feature_folder, "metadata.csv"),
    index=False
)

print("\nFeature extraction completed.")
print("Videos processed:", len(metadata))