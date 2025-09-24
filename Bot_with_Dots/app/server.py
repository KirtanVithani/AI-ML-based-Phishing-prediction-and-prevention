import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import load_model, predict_proba, UrlFeatureModel, save_model
from .llm_explain import generate_explanation
from .data import load_and_merge


load_dotenv()


app = FastAPI(title="Phishing Detection MVP")

# Add a root endpoint to handle GET requests to '/'
@app.get("/")
def read_root():
	return {"message": "Welcome to the Phishing Detection API. Use /predict or /explain endpoints."}


class PredictRequest(BaseModel):
	text: str


class PredictResponse(BaseModel):
	label: str
	confidence: float


class ExplainResponse(BaseModel):
	explanation: str


MODEL_PATH = os.getenv("MODEL_PATH", "models/phish_model.joblib")
_model = None


def _maybe_auto_train() -> None:
	"""Train a model from local CSVs if no saved model exists. Falls back to tiny seed data."""
	root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	urldata = os.path.join(root, "urldata.csv")
	benign = os.path.join(root, "benign_urls.csv")
	phishtank = os.path.join(root, "dataset_phishtank.csv")
	candidates = {"urldata": urldata if os.path.exists(urldata) else None,
				 "benign": benign if os.path.exists(benign) else None,
				 "phishtank": phishtank if os.path.exists(phishtank) else None}
	try:
		if any(candidates.values()):
			df = load_and_merge(candidates["urldata"], candidates["benign"], candidates["phishtank"]) 
			texts = df["url"].tolist()
			labels = df["label"].fillna("benign").astype(str).str.lower().tolist()
			model = UrlFeatureModel()
			model.fit(texts, labels)
			os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
			save_model(model, MODEL_PATH)
			return
	except Exception:
		pass
	# Fallback minimal seed training so API works
	texts = [
		"https://www.google.com/",
		"https://github.com/login",
		"http://paypal.com.secure-update.info/verify",
		"http://192.168.1.10/confirm",
		"https://amazon-secure-check.com/update",
		"http://example.com/",
	]
	labels = [
		"benign",
		"benign",
		"malicious",
		"malicious",
		"malicious",
		"benign",
	]
	model = UrlFeatureModel()
	model.fit(texts, labels)
	os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
	save_model(model, MODEL_PATH)


@app.on_event("startup")
def _startup() -> None:
	global _model
	if not os.path.exists(MODEL_PATH):
		_maybe_auto_train()
	_model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
	if _model is None:
		raise HTTPException(status_code=500, detail="Model not loaded; train first.")
	label, confidence = predict_proba(_model, req.text)
	return PredictResponse(label=label, confidence=confidence)


@app.post("/explain", response_model=ExplainResponse)
def explain(req: PredictRequest):
	if _model is None:
		raise HTTPException(status_code=500, detail="Model not loaded; train first.")
	label, confidence = predict_proba(_model, req.text)
	explanation = generate_explanation(text=req.text, predicted_label=label, confidence=confidence, model=_model)
	return ExplainResponse(explanation=explanation)
