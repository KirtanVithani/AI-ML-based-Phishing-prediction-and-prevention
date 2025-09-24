from __future__ import annotations

import math
import re
from urllib.parse import urlparse
from collections import Counter
import tldextract


SPECIAL_CHARS = set("@-_.?=/#%&:+$!*,'(){}[]|\\")


def shannon_entropy(s: str) -> float:
	if not s:
		return 0.0
	counts = Counter(s)
	n = len(s)
	entropy = 0.0
	for c in counts.values():
		p = c / n
		entropy -= p * math.log2(p)
	return entropy


def count_digits(s: str) -> int:
	return sum(ch.isdigit() for ch in s)


def count_special(s: str) -> int:
	return sum(ch in SPECIAL_CHARS for ch in s)


def has_ip_in_domain(netloc: str) -> int:
	return int(bool(re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", netloc)))


def num_subdomains(extracted) -> int:
	parts = [p for p in extracted.subdomain.split(".") if p]
	return len(parts)


def suspicious_keywords(s: str) -> int:
	keywords = [
		"login",
		"verify",
		"update",
		"secure",
		"account",
		"password",
		"bank",
		"confirm",
		"invoice",
		"paypal",
		"apple",
		"microsoft",
		"amazon",
	]
	low = s.lower()
	return int(any(k in low for k in keywords))


def extract_url_features(text: str) -> dict:
	url = text.strip()
	parsed = urlparse(url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url)
	full = parsed.geturl()
	netloc = parsed.netloc
	path = parsed.path or ""
	query = parsed.query or ""
	fragment = parsed.fragment or ""
	extracted = tldextract.extract(netloc)
	domain = ".".join([p for p in [extracted.domain, extracted.suffix] if p])

	features = {
		"length": len(full),
		"num_digits": count_digits(full),
		"num_special": count_special(full),
		"entropy": shannon_entropy(full),
		"has_ip": has_ip_in_domain(netloc),
		"num_subdomains": num_subdomains(extracted),
		"path_length": len(path),
		"query_length": len(query),
		"fragment_length": len(fragment),
		"suspicious_kw": suspicious_keywords(full),
		"at_sign": int("@" in full),
		"dash_in_domain": int("-" in netloc),
		"https": int(parsed.scheme.lower() == "https"),
	}
	return features


FEATURE_ORDER = [
	"length",
	"num_digits",
	"num_special",
	"entropy",
	"has_ip",
	"num_subdomains",
	"path_length",
	"query_length",
	"fragment_length",
	"suspicious_kw",
	"at_sign",
	"dash_in_domain",
	"https",
]


def features_to_vector(features: dict) -> list[float]:
	return [float(features[name]) for name in FEATURE_ORDER]
