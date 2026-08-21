"""
Out-of-distribution (OOD) detection: flag queries that don't look like a
financial complaint at all, instead of forcing them into one of the known
categories with false confidence.

Same two-signal approach as the Banking77 version of this project:
  1. Softmax/confidence threshold on the classifier's top prediction.
  2. Embedding-distance: cosine distance from the query's sentence embedding
     to the nearest training-set centroid (one centroid per category).

This module is dataset-agnostic on purpose -- it doesn't know or care
whether "categories" means Banking77 intents or CFPB products.
"""
from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast sentence embedding model


class OODDetector:
    def __init__(self, confidence_threshold=0.35, distance_threshold=0.55):
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.centroids = None
        self.centroid_labels = None

    def fit(self, train_texts, train_labels):
        """Build one centroid per category from the training set embeddings."""
        embeddings = self.embedder.encode(list(train_texts), show_progress_bar=True, normalize_embeddings=True)
        labels = np.array(train_labels)
        unique_labels = sorted(set(labels))
        centroids = [embeddings[labels == lab].mean(axis=0) for lab in unique_labels]
        self.centroids = np.vstack(centroids)
        self.centroid_labels = unique_labels
        self.centroids = self.centroids / np.linalg.norm(self.centroids, axis=1, keepdims=True)
        return self

    def nearest_centroid_distance(self, text):
        emb = self.embedder.encode([text], normalize_embeddings=True)[0]
        sims = self.centroids @ emb
        best_idx = sims.argmax()
        return 1 - sims[best_idx], self.centroid_labels[best_idx]

    def is_ood(self, text, top_softmax_confidence):
        dist, nearest_label = self.nearest_centroid_distance(text)
        low_conf = top_softmax_confidence < self.confidence_threshold
        far_from_known = dist > self.distance_threshold

        details = {
            "softmax_confidence": float(top_softmax_confidence),
            "nearest_centroid_distance": float(dist),
            "nearest_label": nearest_label,
        }
        if low_conf and far_from_known:
            return True, "low confidence AND far from all known clusters", details
        elif far_from_known:
            return True, "far from all known clusters", details
        elif low_conf:
            return True, "low classifier confidence", details
        return False, "in-distribution", details

    def save(self, path=None):
        path = path or (MODEL_DIR / "ood_detector.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "confidence_threshold": self.confidence_threshold,
                "distance_threshold": self.distance_threshold,
                "centroids": self.centroids,
                "centroid_labels": self.centroid_labels,
            }, f)

    @classmethod
    def load(cls, path=None):
        path = path or (MODEL_DIR / "ood_detector.pkl")
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(state["confidence_threshold"], state["distance_threshold"])
        obj.centroids = state["centroids"]
        obj.centroid_labels = state["centroid_labels"]
        return obj