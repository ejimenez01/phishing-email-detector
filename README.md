# Phishing Email Detection System

An AI-powered phishing email classifier combining classical machine learning with LLM-generated plain-English explanations. Deployable as a REST API or interactive web app.

**[🚀 Live Demo](https://huggingface.co/spaces/ejimenez01/phishing-email-detector)** · **[API Docs](http://localhost:8000/docs)**

---

## Results — Model Benchmark

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | ~97.2% | ~95.6% | ~97.4% | ~96.5% |
| Complement Naive Bayes | — | — | — | — |
| Linear SVM | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

> Run `python src/train.py` to populate the benchmark table with your results.

![Model Benchmark](models/benchmark.png)

---

## API Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "email_text": "URGENT: Your PayPal account has been suspended. Click here: http://paypa1.net",
    "include_explanation": true
  }'
```

```json
{
  "label": "Phishing Email",
  "confidence": 0.9741,
  "is_phishing": true,
  "top_indicators": ["urgent", "suspended", "click", "verify", "urltoken"],
  "structural_flags": {
    "url_count": 1,
    "has_urgency": 1,
    "has_action_words": 1,
    "exclamation_count": 0,
    "link_mismatch": 1,
    "dollar_sign": 0
  },
  "explanation": "This email is almost certainly a phishing attempt. It uses classic urgency tactics ('URGENT', 'suspended') combined with a suspicious URL that mimics PayPal but uses a misspelled domain. You should delete this email and never click the link — if you're concerned about your account, go directly to paypal.com by typing it into your browser.",
  "model_name": "XGBoost"
}
```

---

## Architecture

```
Email text
    │
    ├─ TF-IDF (15K features, unigrams + bigrams)
    │
    ├─ Structural features (URL count, urgency words,
    │   caps ratio, link mismatch, exclamation count…)
    │
    └─ ML Classifier (best of 5 models, SMOTE-balanced)
            │
            ├─ label + confidence
            │
            └─ LLM (Claude) → plain-English explanation
```

---

## Technologies

- **ML**: scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **NLP**: TF-IDF with bigrams, structural heuristics
- **API**: FastAPI + Pydantic v2
- **LLM**: Anthropic Claude (explanation layer)
- **Deployment**: Docker, Hugging Face Spaces (Gradio)
- **Monitoring**: MLflow (experiment tracking), Evidently AI (drift detection)
- **Testing**: pytest, GitHub Actions CI

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/ejimenez01/phishing-email-detector
cd phishing-email-detector

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download dataset

Download `Phishing_Email.csv` from [Kaggle](https://www.kaggle.com/datasets/rohit08/phishing-email-dataset) and place it in `data/`.

### 3. Train

```bash
python src/train.py
```

This trains 5 models, applies SMOTE, saves the best model to `models/`, and generates benchmark plots.

### 4. Run the API

```bash
uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

Set your API key for LLM explanations:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run with Docker

```bash
docker-compose up --build
```

### 6. Run the Gradio UI

```bash
python gradio_app.py
# → http://localhost:7860
```

---

## Testing

```bash
pytest tests/ -v
```

Unit tests (no model required): feature extraction, text cleaning, structural flags.
Integration tests (requires trained model): model predictions, API endpoints, input validation.

---

## Monitoring

```bash
# Log training run to MLflow
python monitoring/drift.py --mode log_training

# Check for concept drift in recent predictions
python monitoring/drift.py --mode check_drift

# View MLflow UI
open http://localhost:5001
```

---

## Project Structure

```
phishing-email-detector/
├── data/                    # Dataset (not committed)
├── src/
│   ├── features.py          # TF-IDF + structural feature extraction
│   └── train.py             # Multi-model benchmark + SMOTE training
├── app/
│   ├── main.py              # FastAPI endpoints
│   └── schemas.py           # Pydantic request/response models
├── models/                  # Saved model + vectorizer (not committed)
├── tests/
│   ├── test_model.py        # Unit + model quality tests
│   └── test_api.py          # API integration tests
├── monitoring/
│   └── drift.py             # MLflow logging + Evidently drift detection
├── gradio_app.py            # Hugging Face Spaces UI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Author

**Esteban Jimenez Arias** · [GitHub](https://github.com/ejimenez01)
