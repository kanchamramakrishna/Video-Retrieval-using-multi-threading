# Video Retrieval

A content-based video retrieval prototype. It represents each video frame with
color histograms, compares query frames with frames in each database video, and
ranks the database videos by their similarity to the query.

## Retrieval techniques

1. **Frame-level color features:** Each frame is converted from BGR to RGB.
   The red, green, and blue channels are each represented by an independently
   normalized 8-bin histogram. Concatenating the histograms gives a
   24-dimensional feature vector per frame.
2. **Histogram correlation:** For each query-frame/database-frame pair,
   OpenCV's histogram correlation (`HISTCMP_CORREL`) measures similarity.
3. **Best-frame matching and video scoring:** For each query frame, the
   highest correlation against any frame in a database video is selected.
   The mean of those per-query-frame best scores is the video's similarity.
4. **Ranking:** Videos are sorted by descending similarity. The top five are
   printed, and all ranked results are saved to a CSV file.

This is an exhaustive frame-to-frame comparison using color features; it does
not use a learned visual model or an approximate-nearest-neighbor index.

## Data parallelism

The workload is naturally data-parallel: videos can be feature-extracted
independently, and each database video's retrieval score can be computed
independently. These independent tasks can be assigned to separate threads
(multithreading as data parallelism). **The current scripts process these
loops sequentially; multithreading is not implemented in the code yet.**

## Project files

- `feature_extraction.py` — extracts and saves features for training videos,
  plus a `metadata.csv` index.
- `query_processing.py` — extracts frame features from one query video.
- `retrieval.py` — scores and ranks training videos against the saved query
  features.
- `data/` — sample video files.

## Requirements

Python 3 and the following packages:

```powershell
python -m pip install opencv-python numpy pandas
```

## Prepare paths and run

The scripts currently contain absolute Windows paths. Edit the path variables
in each script to match your local dataset before running them. The expected
dataset organization is:

```text
dataset/
  train/       # training .avi videos
  query/       # query .avi video
  features/    # generated training features and metadata
  results/     # generated retrieval results
```

Run the scripts from the project directory in this order:

```powershell
python feature_extraction.py
python query_processing.py
python retrieval.py
```

Set the training, feature, and query paths consistently across the scripts.
The query-processing script writes `query_features.npy`; retrieval reads that
file along with the training feature files and `metadata.csv`. Retrieval
results are written to `results/retrieval_results.csv`.
