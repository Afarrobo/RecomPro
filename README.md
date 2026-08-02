# RecomPro

**RecomPro** is an aspect-aware multilingual product recommendation system built using **FastAPI**, **PostgreSQL**, and **React (Vite)**.

The system imports product reviews from a multi-sheet Excel dataset into PostgreSQL, constructs product profiles from review data, and generates recommendations using a hybrid ranking approach that combines:

- Dense semantic retrieval (Multilingual E5)
- Sparse retrieval (TF-IDF)
- Aspect sentiment analysis
- Category matching
- Price-aware ranking

The frontend provides a simple search interface that returns the most relevant products based on user queries in **English, Bangla, and Banglish**.

```
recompro/
│
├── backend/     FastAPI API, PostgreSQL integration, Recommendation Engine
└── frontend/    React (Vite) User Interface
```

---

# System Workflow

The recommendation pipeline consists of the following stages:

1. Product reviews are imported from a multi-sheet Excel workbook into PostgreSQL.
2. Reviews belonging to the same product are grouped together.
3. Product profiles are generated from review text, ratings, metadata, and aspect sentiments.
4. Dense embeddings are generated using the multilingual E5 encoder.
5. A TF-IDF index is created for sparse retrieval.
6. User queries are analyzed to identify:
   - semantic meaning
   - product category
   - aspect preferences
   - price constraints
7. Multiple ranking signals are combined into a final recommendation score.
8. The top-ranked products are returned through the FastAPI API and displayed in the React frontend.

---

# 1. Create the PostgreSQL Database

### Using pgAdmin

1. Open pgAdmin and connect to the PostgreSQL server.
2. Create a Login Role.

```
Name:
recompro_user
```

Enable:

```
Can Login
```

3. Create a new database.

```
Database:
recompro

Owner:
recompro_user
```

Database tables are created automatically during data loading.

---

### SQL Alternative

```sql
CREATE ROLE recompro_user
WITH LOGIN PASSWORD 'recompro_pass';

CREATE DATABASE recompro
OWNER recompro_user;
```

---

# 2. Backend Setup

```bash
cd backend

python -m venv .venv

source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
```

Update `.env`

```
DATABASE_URL=postgresql+psycopg2://recompro_user:recompro_pass@localhost:5432/recompro

EXCEL_PATH=/path/to/Product_Reviews.xlsx
```

---

# Import Reviews into PostgreSQL

```bash
python -m scripts.load_excel_to_postgres
```

The loader performs the following operations:

- Reads every worksheet from the Excel dataset.
- Merges all review records into a single `reviews` table.
- Stores common review information in database columns.
- Stores aspect sentiment values as a JSON field because aspect names differ across categories.

Example:

```json
{
  "battery": 1,
  "quality": 1,
  "mic": -1
}
```

Running the loader again refreshes the database by replacing existing review records.

---

# Run the Backend

```bash
uvicorn app.main:app --reload --port 8000
```

During the first startup the backend automatically:

1. Loads review data from PostgreSQL.
2. Builds the product catalog.
3. Generates multilingual product embeddings using the E5 encoder.
4. Creates a TF-IDF index.
5. Stores generated artifacts inside:

```
backend/artifacts/
```

Subsequent startups reuse these cached files, resulting in significantly faster loading.

---

### API Test

```bash
curl "http://localhost:8000/api/recommend?q=wireless%20earbuds%20with%20good%20bass&top_k=5"
```

---

# Using a Fine-Tuned Dense Model (Optional)

If a fine-tuned multilingual E5 model is available (for example, `dense_e5_best/` generated during training), configure the backend to use it by adding:

```
DENSE_MODEL_DIR=/path/to/dense_e5_best
```

Delete the cached artifacts before restarting:

```
backend/artifacts/
```

The catalog embeddings will be regenerated using the fine-tuned model.

---

# 3. Frontend Setup

```bash
cd frontend

npm install

cp .env.example .env

npm run dev
```

Example `.env`

```
VITE_API_URL=http://localhost:8000
```

Open the local development URL displayed in the terminal (typically `http://localhost:5173`).

Example search queries:

- wireless earbuds with good bass
- budget keyboard
- gaming mouse
- কম দামে ভালো পাওয়ার ব্যাংক
- ভালো ক্যামেরার স্মার্টফোন

---

# Recommendation Workflow

When a search request is submitted:

1. The React frontend sends the query to the FastAPI API.
2. The query is encoded using the multilingual E5 model.
3. Dense similarity is calculated against product embeddings.
4. Sparse similarity is computed using the TF-IDF index.
5. Product category is detected.
6. Aspect-related keywords are identified.
7. Price constraints are extracted when available.
8. Individual ranking signals are combined into a hybrid score.
9. Products are ranked based on the final score.
10. The highest-ranked products are returned to the frontend.

The final ranking combines:

- Dense semantic similarity
- Sparse lexical similarity
- Aspect sentiment scores
- Category matching
- Product ratings
- Price-aware scoring

---

# Troubleshooting

### Reviews table not found

Run the data loader:

```bash
python -m scripts.load_excel_to_postgres
```

Also verify:

- `DATABASE_URL`
- `EXCEL_PATH`

---

### CORS Error

Ensure the frontend origin matches the backend CORS configuration.

---

### Slow First Startup

The first launch generates embeddings, builds the TF-IDF index, and creates cached artifacts. Future startups reuse the cached files.

---

### "Failed to Fetch"

Possible causes:

- Backend server is not running.
- Incorrect `VITE_API_URL`.
- Backend API is inaccessible.
