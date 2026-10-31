import cv2
import numpy as np
import os

query_path = r"C:\Users\JAIVIVEKNAGABHYRU\Documents\vscode\dataset\query\v_ShavingBeard_g01_c01.avi"
output_path = r"C:\Users\JAIVIVEKNAGABHYRU\Documents\vscode\dataset\query\query_features.npy"

cap = cv2.VideoCapture(query_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open query video: {query_path}")

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

    features.append(np.concatenate([r_hist, g_hist, b_hist]))

cap.release()

features = np.asarray(features, dtype=np.float32)

if len(features) == 0:
    raise ValueError("No frames were extracted from the query video.")

np.save(output_path, features)

print("Query video:", os.path.basename(query_path))
print("Frames processed:", len(features))
print("Feature shape:", features.shape)
print("Saved to:", output_path)
