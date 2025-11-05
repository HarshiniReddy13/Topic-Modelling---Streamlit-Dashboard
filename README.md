# 🧠 Parallel Text Processor
### Accelerating NLP Workflows with Parallel Computing and Topic Modeling

## 📘 Overview
The **Parallel Text Processor** is a system designed to process large text datasets efficiently using **parallel processing** and **advanced NLP techniques**.  
It compares traditional and modern topic modeling methods — **Latent Dirichlet Allocation (LDA)** and **K-Means with Sentence Embeddings** — and presents results in an **interactive Streamlit dashboard**.

---

## 🚀 Features
- **Parallel Text Preprocessing**: Multi-core tokenization, lemmatization, and stopword removal using SpaCy and Python multiprocessing.  
- **Text Modeling**: Implements both traditional (LDA) and modern (embedding-based K-Means) topic modeling techniques.  
- **Efficient Vectorization**: Uses TF-IDF and CountVectorizer for feature extraction.  
- **Visualization Dashboard**: Interactive Streamlit app comparing K-Means and LDA results.  
- **Performance Metrics**: Includes Silhouette Score, Topic Flow (Sankey Diagram), and Time Comparison charts.  
- **Automatic Topic Labeling**: Uses cosine similarity to assign human-readable topic labels.

---

## 🧩 System Workflow
1. **Upload Dataset** → Upload CSV or text data.  
2. **Preprocessing** → Cleaning, lemmatizing, and tokenizing using SpaCy.  
3. **Embedding Generation** → Semantic sentence embeddings with SentenceTransformer.  
4. **Dimensionality Reduction** → UMAP reduces embeddings for faster clustering.  
5. **Clustering and Topic Modeling** → K-Means and LDA applied in parallel.  
6. **Visualization** → Interactive Streamlit dashboard showing topics, scores, and time efficiency.

---

## ⚙️ Technologies Used
- **Programming Language**: Python  
- **Libraries**: Streamlit, SpaCy, SentenceTransformer, Scikit-learn, UMAP, Matplotlib, Plotly  
- **Parallelism**: Python Multiprocessing module  
- **Visualization**: Plotly, Matplotlib, Streamlit components  

---

## 📊 Results
- Parallel preprocessing improved performance by up to **40–60%** compared to sequential execution.  
- K-Means produced **semantically coherent topics**, while LDA offered **probabilistic interpretability**.  
- Dashboard enables easy comparison of **model performance, topic clarity, and time efficiency**.

---

## 🧠 Key Concepts
- **Parallelism in NLP**: Enables faster text processing by dividing tasks across CPU cores.  
- **Embedding-based Clustering**: Uses transformer embeddings for meaning-driven topic grouping.  
- **Probabilistic Topic Modeling**: LDA identifies latent topics through word distributions.  

