import os
import cv2
import numpy as np
import pandas as pd
base_path = r"C:\Users\JAIVIVEKNAGABHYRU\Documents\vscode"
feature_folder = os.path.join(base_path, "dataset", "features")
query_path = os.path.join(base_path, "dataset", "query", "query_features.npy")
train_folder = os.path.join(base_path, "dataset", "train")
metadata_path = os.path.join(feature_folder, "metadata.csv")
query_features = np.load(query_path)
metadata = pd.read_csv(metadata_path)
results = []
for _, row in metadata.iterrows():
    feature_path = os.path.join(feature_folder, row["feature_file"])
    train_features = np.load(feature_path)
    frame_scores = []
    for query_frame in query_features:
        scores = []
        for train_frame in train_features:
            score = cv2.compareHist(
                query_frame.astype(np.float32),
                train_frame.astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            scores.append(score)
        frame_scores.append(max(scores))
    video_score = np.mean(frame_scores)
    results.append({
        "video": row["video"],
        "similarity": video_score
    })
results.sort(key=lambda x: x["similarity"], reverse=True)
print("\nTop 5 Retrieved Videos")
for result in results[:5]:
    print(result["video"], "Similarity:", round(result["similarity"], 4))
output_path = os.path.join(base_path, "results", "retrieval_results.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
pd.DataFrame(results).to_csv(output_path, index=False)
print("\nResults saved to:", output_path)
