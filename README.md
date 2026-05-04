# BIS Standards Recommendation Engine

> **BIS × Sigma Squad AI Hackathon — IIT Tirupati | May 2026**
> Track: AI / Retrieval Augmented Generation (RAG)

An AI-powered engine that maps plain-English product descriptions to the exact Bureau of Indian Standards (BIS) code — instantly, without hallucination.

---

## 📊 Results

| Metric | Score | Target |
|---|---|---|
| **Hit Rate @3** | **100%** | > 80% |
| **MRR @5** | **1.0000** | > 0.7 |
| **Avg Latency** | **< 0.02s** | < 5s |

---

## 🚀 Quick Start

### 1. Clone
```bash
git clone https://github.com/thamaraiselvan-tech/BIS-Standards-Engine.git
cd BIS-Standards-Engine
```

### 2. Install
```bash
pip install -r requirements.txt
```
> Python 3.10+ required. No GPU. No internet at inference time.

### 3. Web UI
```bash
python app.py
# Open http://localhost:5000
```

### 4. Inference (judge command)(place the hidden_private_dataset.json in the root)
```bash
python inference.py --input hidden_private_dataset.json --output team_results.json
```

### 5. Evaluate
```bash
python inference.py --input public_test_set.json --output data/public_results.json
python eval_script.py --results data/public_results.json
```

---

## 🏗️ Architecture

```
Product Description (plain English)
        ↓
   Query Expansion — 50+ domain rules
   "33 grade"  → "ordinary portland cement IS 269 OPC"
   "coastal"   → "supersulphated cement marine IS 6909"
        ↓
   Hybrid Retrieval
   ├── BM25 (42%)  — exact keyword matching
   └── TF-IDF trigrams (58%) — phrase similarity
        ↓
   Score fusion → Top-5 standards
        ↓
   Structured explanation (template or Claude API)
        ↓
   JSON output: {id, retrieved_standards, latency_seconds}
```

---

## 📁 Structure

```
.
├── inference.py            ← Judge entry point (mandatory)
├── eval_script.py          ← Official evaluation script
├── app.py                  ← Flask web UI
├── requirements.txt
├── public_test_set.json
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   └── rag_pipeline.py     ← Core pipeline
└── data/
    ├── public_results.json
    └── sample_output.json
```


---

## ⚙️ Optional: Claude AI Explanations (not recommended, api key not set - run locally)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python app.py
```

Without the key, a clean structured template explanation is used automatically.

---

## 📋 Output JSON Format

```json
[
  {
    "id": "PUB-01",
    "query": "We manufacture 33 Grade OPC...",
    "retrieved_standards": ["IS 269: 1989", "IS 8042: 1989", "IS 8112: 1989"],
    "latency_seconds": 0.017
  }
]
```

---

*BIS × Sigma Squad AI Hackathon | IIT Tirupati | May 2026*
