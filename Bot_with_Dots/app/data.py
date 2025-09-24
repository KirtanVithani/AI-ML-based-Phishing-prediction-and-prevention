from __future__ import annotations

import pandas as pd
from typing import Optional, List
import re


def _looks_like_url(value: str) -> bool:
	if not isinstance(value, str):
		return False
	text = value.strip().lower()
	if not text or " " in text:
		return False
	if text.startswith(("http://", "https://")):
		return True
	# domain.tld pattern
	return bool(re.search(r"[a-z0-9.-]+\.[a-z]{2,}(/|$)", text))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
	cols = {c.lower().strip(): c for c in df.columns}
	df = df.rename(columns=cols)
	# Try to identify URL and label columns
	url_col = None
	for candidate in [
		"url",
		"urls",
		"phishing_url",
		"link",
		"links",
		"hostname",
		"domain",
		"website",
		"entry",
	]:
		if candidate in df.columns:
			url_col = candidate
			break
	# Heuristic fallback: pick the column with most URL-like values
	if url_col is None:
		best_col = None
		best_score = 0
		for c in df.columns:
			try:
				series = df[c].astype(str)
			except Exception:
				continue
			match_ratio = series.map(_looks_like_url).mean()
			if match_ratio > best_score:
				best_score = match_ratio
				best_col = c
		if best_col is not None and best_score >= 0.2:
			url_col = best_col
	if url_col is None:
		raise ValueError("Could not find a URL column in dataset")
	# Standardize to 'url'
	if url_col != "url":
		df = df.rename(columns={url_col: "url"})
	# Standardize label to 'label' with values in {malicious, benign}
	label_col = None
	for candidate in ["label", "type", "is_phishing", "target", "class", "status", "result"]:
		if candidate in df.columns:
			label_col = candidate
			break
	if label_col is not None and label_col != "label":
		df = df.rename(columns={label_col: "label"})
	# Normalize label values if present
	if "label" in df.columns:
		df["label"] = (
			df["label"].astype(str).str.lower().str.strip()
			.replace({
				"phishing": "malicious",
				"malware": "malicious",
				"bad": "malicious",
				"1": "malicious",
				"true": "malicious",
				"benign": "benign",
				"good": "benign",
				"0": "benign",
				"false": "benign",
			})
		)
		# default others to benign
		df.loc[~df["label"].isin(["malicious", "benign"]), "label"] = "benign"
	else:
		# If no label, default to unknown; upstream can set
		df["label"] = pd.NA
	# Clean url
	df["url"] = df["url"].astype(str).str.strip()
	return df[["url", "label"]].dropna(subset=["url"]).drop_duplicates()


def load_and_merge(
	urldata: Optional[str] = None,
	benign_urls: Optional[str] = None,
	phishtank: Optional[str] = None,
	default_benign_label: str = "benign",
	default_malicious_label: str = "malicious",
) -> pd.DataFrame:
	frames: List[pd.DataFrame] = []
	for path in [urldata, benign_urls, phishtank]:
		if path:
			# Try default, then headerless
			try:
				df = pd.read_csv(path)
			except Exception:
				df = pd.read_csv(path, header=None)
				# Create a generic column name for single-column csvs
				if df.shape[1] == 1:
					df = df.rename(columns={df.columns[0]: "url"})
			df = _normalize_columns(df)
			frames.append(df)
	if not frames:
		raise ValueError("No datasets provided.")
	merged = pd.concat(frames, ignore_index=True)
	# If any unlabeled rows, attempt heuristic: default to benign
	if merged["label"].isna().any():
		merged.loc[merged["label"].isna(), "label"] = default_benign_label
	# Ensure only two classes
	merged.loc[~merged["label"].isin([default_benign_label, default_malicious_label]), "label"] = default_benign_label
	merged = merged.dropna(subset=["url"]).drop_duplicates(subset=["url"]).reset_index(drop=True)
	return merged
