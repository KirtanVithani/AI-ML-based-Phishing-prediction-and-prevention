from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv

from .features import extract_url_features

load_dotenv()


def _heuristic_explanation(text: str, predicted_label: str, confidence: float) -> str:
	f = extract_url_features(text)
	reasons = []
	if f["suspicious_kw"]:
		reasons.append("contains suspicious keywords")
	if f["has_ip"]:
		reasons.append("uses raw IP in domain")
	if f["num_subdomains"] > 2:
		reasons.append("many subdomains")
	if f["dash_in_domain"]:
		reasons.append("dash in domain")
	if f["at_sign"]:
		reasons.append("contains '@'")
	if f["entropy"] > 4.0:
		reasons.append("high character entropy")
	if f["num_special"] > 5:
		reasons.append("many special characters")
	if f["length"] > 80:
		reasons.append("unusually long URL")
	if not reasons:
		reasons.append("overall pattern resembles known benign/malicious distributions")
	return f"Model predicts {predicted_label} with confidence {confidence:.2f} because it {', '.join(reasons)}."


def generate_explanation(text: str, predicted_label: str, confidence: float, model=None) -> str:
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		return _heuristic_explanation(text, predicted_label, confidence)
	try:
		from openai import OpenAI
		client = OpenAI(api_key=api_key)
		prompt = (
			"You are a security assistant. Explain in 2-4 concise sentences why the following URL or email "
			"might be phishing or benign. Mention key indicators (keywords, domain patterns, entropy, length, IP usage).\n"
			f"Text: {text}\n"
			f"Model prediction: {predicted_label} (confidence {confidence:.2f})\n"
		)
		resp = client.chat.completions.create(
			model="gpt-4o-mini",
			messages=[{"role": "user", "content": prompt}],
			temperature=0.4,
		)
		return resp.choices[0].message.content.strip()
	except Exception:
		return _heuristic_explanation(text, predicted_label, confidence)
