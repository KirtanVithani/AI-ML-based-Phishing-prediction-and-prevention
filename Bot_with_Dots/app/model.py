from __future__ import annotations

from typing import Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .features import extract_url_features, features_to_vector, FEATURE_ORDER


class UrlFeatureModel:
	"""A minimal wrapper combining feature extraction and a classifier."""

	def __init__(self, classifier=None):
		self.classifier = classifier or RandomForestClassifier(n_estimators=200, random_state=42)
		self.feature_names = FEATURE_ORDER

	def fit(self, texts, labels):
		X = np.array([features_to_vector(extract_url_features(t)) for t in texts], dtype=float)
		y = np.array(labels)
		self.classifier.fit(X, y)
		return self

	def predict_proba(self, text: str) -> Tuple[str, float]:
		X = np.array([features_to_vector(extract_url_features(text))], dtype=float)
		probas = self.classifier.predict_proba(X)[0]
		classes = list(self.classifier.classes_)
		# Ensure 'malicious' probability returned
		if "malicious" in classes:
			idx = classes.index("malicious")
			p_mal = float(probas[idx])
			label = "malicious" if p_mal >= 0.5 else "benign"
			confidence = p_mal if label == "malicious" else 1.0 - p_mal
		else:
			# Fallback if classes unexpected
			idx_max = int(np.argmax(probas))
			label = classes[idx_max]
			confidence = float(probas[idx_max])
		return label, confidence


def save_model(model: UrlFeatureModel, path: str) -> None:
	joblib.dump(model, path)


def load_model(path: str) -> UrlFeatureModel:
	return joblib.load(path)


def predict_proba(model: UrlFeatureModel, text: str) -> Tuple[str, float]:
	return model.predict_proba(text)
