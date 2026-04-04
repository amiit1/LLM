# NLP Preprocessing App

A clean, modular, and beginner-friendly NLP preprocessing project with:

- FastAPI backend
- spaCy + NLTK NLP pipeline
- Interactive Tailwind + Vanilla JavaScript frontend

## Features

- Tokenization
- Lemmatization
- Stemming
- Part-of-Speech (POS) Tagging
- Named Entity Recognition (NER)
- Combined `/analyze` endpoint for all tasks
- Interactive UI with loading state, transitions, entity highlighting, sample text, and copy-to-clipboard

## Project Structure

```text
nlp-preprocessing-app/
|
+-- backend/
|   +-- app.py
|   +-- routes/
|   |   +-- nlp_routes.py
|   +-- services/
|   |   +-- nlp_service.py
|   +-- utils/
|   |   +-- helpers.py
|   +-- models/
|       +-- schemas.py
|
+-- frontend/
|   +-- index.html
|   +-- script.js
|   +-- styles.css
|
+-- examples/
|   +-- lemma_vs_stem.md
|
+-- requirements.txt
+-- README.md
```

## Installation

1. Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Install the spaCy English model.

```bash
python -m spacy download en_core_web_sm
```

## Run Backend

From the `backend` folder:

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Backend URL:

- API root: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

## Run Frontend

Open `frontend/index.html` in your browser.

Recommended (to avoid local-file restrictions):

```bash
cd frontend
python -m http.server 5500
```

Then open:

- `http://127.0.0.1:5500`

## API Endpoints

All endpoints are under `/api/nlp` and accept:

```json
{
  "text": "Apple is opening a new office in Bengaluru."
}
```

Available routes:

- `POST /api/nlp/tokenize`
- `POST /api/nlp/lemmatize`
- `POST /api/nlp/stem`
- `POST /api/nlp/pos-tag`
- `POST /api/nlp/ner`
- `POST /api/nlp/analyze`

### Example cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/nlp/analyze" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Microsoft hired Alex in Seattle in 2024.\"}"
```

## Notes for Beginners

- Service functions in `backend/services/nlp_service.py` are independent and reusable.
- Routes in `backend/routes/nlp_routes.py` stay lightweight and only orchestrate request/response handling.
- Pydantic schemas in `backend/models/schemas.py` keep API contracts explicit.

## How Preprocessing Works (Step-by-Step)

When you send text to any API endpoint, the processing flow is:

1. **Request validation**

- FastAPI validates the request body using `TextRequest` in `backend/models/schemas.py`.

2. **Route handling**

- Endpoint handlers in `backend/routes/nlp_routes.py` receive the request and call the correct NLP service function.

3. **Text normalization**

- The input text is cleaned by `normalize_whitespace` in `backend/utils/helpers.py`.
- This removes repeated spaces/newlines so outputs are more consistent.

4. **NLP processing**

- Functions in `backend/services/nlp_service.py` run spaCy/NLTK logic and return structured output.

5. **Typed response**

- FastAPI returns response objects defined in `backend/models/schemas.py`.

## Are Pre-Trained Models Used?

Yes.

- The app loads spaCy's pre-trained English model: `en_core_web_sm`.
- This model powers tokenization, POS tagging, lemmatization, and NER.
- The loaded spaCy pipeline typically includes components like:
  - `tok2vec`
  - `tagger`
  - `parser`
  - `attribute_ruler`
  - `lemmatizer`
  - `ner`

Stemming is handled separately using NLTK's **PorterStemmer**, which is rule-based and not a pre-trained statistical model.

## Where Each NLP Option Is Implemented

Core logic file: `backend/services/nlp_service.py`

| Feature       | Function          | Description                                |
| ------------- | ----------------- | ------------------------------------------ |
| Tokenization  | `tokenize(text)`  | Splits text into tokens using spaCy        |
| Lemmatization | `lemmatize(text)` | Returns dictionary base form (`lemma_`)    |
| Stemming      | `stem(text)`      | Applies NLTK Porter stemming rules         |
| POS Tagging   | `pos_tag(text)`   | Returns token + POS/tag details            |
| NER           | `ner(text)`       | Extracts named entities and labels         |
| Analyze All   | `analyze(text)`   | Runs all tasks and returns combined output |

Endpoint mapping file: `backend/routes/nlp_routes.py`

- `POST /api/nlp/tokenize` -> `tokenize(text)`
- `POST /api/nlp/lemmatize` -> `lemmatize(text)`
- `POST /api/nlp/stem` -> `stem(text)`
- `POST /api/nlp/pos-tag` -> `pos_tag(text)`
- `POST /api/nlp/ner` -> `ner(text)`
- `POST /api/nlp/analyze` -> `analyze(text)`

## Why `analyze` Is Efficient

The `analyze(text)` function parses the text once and then derives tokens, lemmas, stems, POS tags, and entities from the same parsed document. This avoids repeated model calls and is better for performance when you need all outputs together.
