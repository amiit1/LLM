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
- Corpus embeddings with TF-IDF on a small custom corpus
- Nearest-neighbor corpus similarity lookup
- Embedding visualization using PCA or t-SNE
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
|   |   +-- embedding_routes.py
|   +-- services/
|   |   +-- nlp_service.py
|   |   +-- embedding_service.py
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

Embedding routes:

- `GET /api/embeddings/info`
- `GET /api/embeddings/corpus?limit=80`
- `POST /api/embeddings/vector`
- `GET /api/embeddings/visualize?method=pca&limit=40`

### Example cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/nlp/analyze" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Microsoft hired Alex in Seattle in 2024.\"}"
```

Embedding vector example:

```bash
curl -X POST "http://127.0.0.1:8000/api/embeddings/vector" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"language models and NLP workflows\", \"top_k\": 5}"
```

## Corpus Embeddings Module

The app now includes an additional section called **Embedding Explorer** in the same frontend page.

What it does:

- Builds TF-IDF-based corpus/document embeddings from a small custom corpus.
- Returns embedding vectors for input text queries.
- Computes nearest corpus entries with cosine similarity.
- Projects corpus embeddings to 2D for visualization with PCA or t-SNE.

Backend implementation:

- Service logic: `backend/services/embedding_service.py`
- API routes: `backend/routes/embedding_routes.py`
- Request/response models: `backend/models/schemas.py`

Frontend implementation:

- New Embedding Explorer section: `frontend/index.html`
- API calls and rendering logic: `frontend/script.js`
- Visualization styling: `frontend/styles.css`

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

## Embeddings Deep Dive (Concepts + Implementation)

This section explains the corpus embedding system in detail: what each term means, how values are computed, and whether the implementation uses explicit logic or inbuilt library calls.

### 1. Corpus Source (Are docs pre-stored?)

Yes. The corpus entries are currently pre-stored in code as a Python list called `CUSTOM_CORPUS`.

- Location: `backend/services/embedding_service.py`
- Current behavior: static in-memory corpus, no database/file loading.
- API access: `GET /api/embeddings/corpus` returns these documents with `doc_id`.

### 2. What TF-IDF Means

TF-IDF stands for **Term Frequency - Inverse Document Frequency**. It converts text documents into numeric vectors where each dimension corresponds to one vocabulary term.

- **TF (Term Frequency)**: how much a term appears in a document.
- **IDF (Inverse Document Frequency)**: how rare or informative that term is across all documents.
- **TF-IDF weight**: higher for terms that are frequent in a document but not common everywhere.

Typical intuition formula:

$$
\mathrm{TF\text{-}IDF}(t,d) = \mathrm{TF}(t,d) \times \mathrm{IDF}(t)
$$

One common IDF variant (used conceptually by scikit-learn with smoothing):

$$
\mathrm{IDF}(t) = \log\left(\frac{1 + N}{1 + \mathrm{df}(t)}\right) + 1
$$

Where:

- $N$ = total number of documents in corpus
- $\mathrm{df}(t)$ = number of documents containing term $t$

### 3. How TF-IDF Is Computed in This Codebase

In `embedding_service.py`, TF-IDF is computed using the inbuilt class `TfidfVectorizer` from scikit-learn.

- Inbuilt call:
  - `vectorizer = TfidfVectorizer(...)`
  - `doc_term_matrix = vectorizer.fit_transform(normalized_docs)`
- `fit_transform(...)` both:
  - learns the vocabulary and IDF values from corpus
  - transforms each corpus document into a TF-IDF vector

Important implementation details:

- Lowercasing is enabled (`lowercase=True`).
- Token pattern is explicitly set (`token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"`), which keeps alphabetic tokens with length >= 2.
- Sparse matrix is converted to dense NumPy array with:
  - `doc_vectors = doc_term_matrix.toarray().astype(np.float32)`

So this part is mainly **inbuilt vectorization** with a few **explicit configuration choices**.

### 4. Query Embedding (How input text is embedded)

For a new user query text:

1. Normalize text with `normalize_whitespace(...).lower()`
2. Transform using the already-fitted vectorizer:
   - `query_sparse = store["vectorizer"].transform([normalized])`
3. Convert to dense vector:
   - `query_vector = query_sparse.toarray()[0]`

This uses **inbuilt transform** for feature extraction and **explicit conversion/validation** in Python.

### 5. Similarity Computation Between Query and Docs

Similarity uses **cosine similarity** (explicitly implemented with NumPy, not via sklearn helper).

Formula used:

$$
\cos(\theta) = \frac{a \cdot b}{\lVert a \rVert \lVert b \rVert}
$$

Code-level computation in `_cosine_similarity(...)`:

- Dot products: `np.dot(matrix, query_vector)`
- Norms: `np.linalg.norm(...)`
- Numerical safety: denominator includes `+ 1e-12`

Nearest neighbors are then selected by sorting scores descending:

- `sorted_indices = np.argsort(scores)[::-1]`
- top `k` entries are returned with `doc_id`, `text`, and `score`.

This part is mostly **explicit custom logic**.

### 6. 2D Visualization (PCA and t-SNE)

The API can project high-dimensional document vectors to 2D for plotting.

- PCA endpoint option: `method=pca`
- t-SNE endpoint option: `method=tsne`

How implemented:

- Select top corpus docs by a simple importance heuristic:
  - `strengths = doc_vectors.sum(axis=1)`
- Build matrix of selected docs.
- Use inbuilt reducers from scikit-learn:
  - `PCA(n_components=2, random_state=42)`
  - `TSNE(n_components=2, random_state=42, ...)`
- Run projection:
  - `reduced = reducer.fit_transform(matrix)`

So dimensionality reduction is **inbuilt**, while selection/ranking and response shaping are **explicit**.

### 7. Inbuilt vs Explicit: Quick Summary

| Step                 | Inbuilt Function/Class          | Explicit Custom Logic                          |
| -------------------- | ------------------------------- | ---------------------------------------------- |
| Corpus vectorization | `TfidfVectorizer.fit_transform` | corpus normalization + token settings          |
| Query vectorization  | `TfidfVectorizer.transform`     | validation and error handling                  |
| Similarity           | none (manual NumPy)             | cosine math, sorting, top-k selection          |
| 2D reduction         | `PCA`, `TSNE`                   | doc ranking (`strengths`) and point formatting |

### 8. Endpoint to Function Mapping (Embeddings)

- `GET /api/embeddings/info` -> `get_embedding_info()`
- `GET /api/embeddings/corpus` -> `get_corpus(limit)`
- `POST /api/embeddings/vector` -> `get_corpus_embedding(text, top_k)`
- `GET /api/embeddings/visualize` -> `project_embeddings(method, limit)`

These are routed in `backend/routes/embedding_routes.py` and implemented in `backend/services/embedding_service.py`.
