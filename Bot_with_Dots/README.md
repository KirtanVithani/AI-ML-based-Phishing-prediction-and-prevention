## Phishing Detection MVP (Bot_with_Dots)

Minimal end-to-end AI/ML phishing URL detector with FastAPI and optional LLM-based explanations.

### Features
- Train a simple ML model (RandomForest) on merged CSV datasets
- URL feature extraction (length, entropy, special chars, IP usage, etc.)
- REST API:
  - POST `/predict` → classification + confidence
  - POST `/explain` → LLM/heuristic explanation
- Model persisted for instant predictions

### Setup
1. Create and activate a virtual environment (Windows PowerShell):
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Copy environment file and set variables:
```powershell
copy .env.example .env
# Edit .env to add OPENAI_API_KEY if you want LLM explanations
```

### Datasets
Place your CSVs anywhere and pass their paths during training. Expected files (examples):
- `urldata.csv`
- `benign_urls.csv`
- `dataset_phishtank.csv`

Each should contain at least a URL column, and optionally a label (`malicious`/`benign`). The training script can infer/normalize columns.

### Train the Model
Run training by pointing to your CSVs:
```powershell
python -m app.train --urldata path\to\urldata.csv --benign_urls path\to\benign_urls.csv --phishtank path\to\dataset_phishtank.csv --model-path models\phish_model.joblib
```

Notes:
- Any of the dataset args are optional; provide what you have.
- Output includes a saved model and a JSON of feature names for consistent inference.

### Start the API Server
```powershell
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

### API Examples
- Predict:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "http://examp1e-login.com/verify?id=123"}'
```
Response:
```json
{"label":"malicious","confidence":0.94}
```

- Explain (LLM if configured, else heuristic):
```bash
curl -X POST http://localhost:8000/explain -H "Content-Type: application/json" -d '{"text": "http://examp1e-login.com/verify?id=123"}'
```

### Environment
See `.env.example` for configurable variables:
- `OPENAI_API_KEY` (optional, enables LLM explanations)
- `MODEL_PATH` (optional; default `models/phish_model.joblib`)

### Project Structure
```
Bot_with_Dots/
  app/
    __init__.py
    server.py
    data.py
    features.py
    llm_explain.py
    model.py
    train.py
  models/
  requirements.txt
  README.md
  .env.example
```

### Notes
- This is an MVP. For production, add robust validation, auth, logging, metrics, tests, and CI.
