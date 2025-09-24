from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from .data import load_and_merge
from .model import UrlFeatureModel, save_model


def _safe_train_test_split(texts, labels, test_size=0.2, random_state=42):
	counts = Counter(labels)
	num_samples = len(labels)
	# Disable stratify if only one class or any class too small for the split
	min_class = min(counts.values()) if counts else 0
	use_stratify = len(counts) > 1 and min_class >= 2 and num_samples >= 5
	try:
		return train_test_split(
			texts,
			labels,
			test_size=test_size,
			random_state=random_state,
			stratify=labels if use_stratify else None,
		)
	except Exception:
		# Fallback: no stratification
		return train_test_split(
			texts,
			labels,
			test_size=test_size,
			random_state=random_state,
			stratify=None,
		)


def main():
	parser = argparse.ArgumentParser(description="Train phishing URL detector")
	parser.add_argument("--urldata", type=str, default=None)
	parser.add_argument("--benign_urls", type=str, default=None)
	parser.add_argument("--phishtank", type=str, default=None)
	parser.add_argument("--model-path", type=str, default="models/phish_model.joblib")
	args = parser.parse_args()

	df = load_and_merge(args.urldata, args.benign_urls, args.phishtank)
	# Ensure labels exist; assume unlabeled as benign
	df["label"] = df["label"].fillna("benign").astype(str).str.lower()
	# Keep only two classes
	df.loc[~df["label"].isin(["benign", "malicious"]), "label"] = "benign"

	texts = df["url"].tolist()
	labels = df["label"].tolist()

	# Handle very small datasets: skip holdout if too few samples
	if len(labels) < 5 or len(set(labels)) < 2:
		X_train, X_test, y_train, y_test = texts, [], labels, []
	else:
		X_train, X_test, y_train, y_test = _safe_train_test_split(texts, labels, test_size=0.2, random_state=42)

	model = UrlFeatureModel()
	model.fit(X_train, y_train)

	# Evaluate if we have a test split
	if X_test:
		y_pred = [model.predict_proba(x)[0] for x in X_test]
		acc = accuracy_score(y_test, y_pred)
		print(f"Holdout accuracy: {acc*100:.2f}% ({acc:.4f})")
		print("Classification report:\n" + classification_report(y_test, y_pred, digits=4))
	else:
		print("Dataset too small or single-class; skipping holdout evaluation.")

	# Save trained model on full data for serving
	model.fit(texts, labels)
	out_path = Path(args.model_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	save_model(model, str(out_path))
	print(f"Saved model to {out_path}")


if __name__ == "__main__":
	main()
