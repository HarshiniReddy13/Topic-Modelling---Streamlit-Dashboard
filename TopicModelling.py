import streamlit as st
from collections import Counter
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
import umap
import nltk
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import plotly.graph_objects as go

nltk.download("stopwords")

# --- Load SpaCy ---
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def process_text_chunk(text_chunk):
    doc = nlp(text_chunk)
    return Counter([token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop])

# --- Load SentenceTransformer model globally ---
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = get_model()

# --- Preprocess Texts ---
@st.cache_data
def preprocess_texts(df, text_col, n_rows):
    df["Article_Clean"] = (
        df[text_col]
        .astype(str)
        .str.lower()
        .str.replace("\xa0", " ")
        .str.replace(r"[^\w\s]", "", regex=True)
    )
    texts = df["Article_Clean"].head(n_rows).tolist()
    subset_df = df.head(n_rows).copy()
    word_counts_list = []
    for text in tqdm(subset_df["Article_Clean"], desc="Processing"):
        word_counts_list.append(process_text_chunk(text))
    subset_df["word_counts"] = word_counts_list
    return texts, subset_df

# --- Embeddings ---
@st.cache_resource
def generate_embeddings(texts):
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

# --- UMAP Reduction ---
@st.cache_resource
def reduce_dimensions(embeddings):
    reducer = umap.UMAP(n_neighbors=15, n_components=50, metric='cosine', random_state=42)
    X_reduced = reducer.fit_transform(embeddings)
    return X_reduced

# --- KMeans Clustering ---
@st.cache_resource
def cluster_texts(X_reduced, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_reduced)
    return clusters

# --- Top Words for KMeans ---
# @st.cache_data
def extract_top_words(df_result, n_clusters, top_n=20, _model=None):
    combined_stopwords = set(ENGLISH_STOP_WORDS).union(set(stopwords.words("english")))
    topics = []

    candidate_labels = [
        "Politics", "Economy", "Sports", "Technology", "Environment",
        "Health", "Education", "Crime", "Entertainment", "World"
    ]
    label_embeddings = _model.encode(candidate_labels)

    for cluster_id in range(n_clusters):
        cluster_texts = " ".join(df_result[df_result["Cluster"] == cluster_id]["Text"].tolist()).lower()
        words = [w for w in cluster_texts.split() if w.isalpha()]

        word_freq = Counter(words)
        auto_stopwords = {w for w, _ in word_freq.most_common(top_n)}
        final_stopwords = combined_stopwords.union(auto_stopwords)

        vectorizer = TfidfVectorizer(max_features=15, stop_words=list(final_stopwords))
        X = vectorizer.fit_transform([cluster_texts])
        top_words = vectorizer.get_feature_names_out()

        cluster_embedding = _model.encode([" ".join(top_words)]).mean(axis=0)
        sims = cosine_similarity([cluster_embedding], label_embeddings)[0]
        best_label = candidate_labels[int(np.argmax(sims))]

        topics.append({
            "Topic Label": f"{best_label}",
            "Top Keywords": ", ".join(top_words)
        })

    return pd.DataFrame(topics)

# --- LDA Topic Modeling ---
# @st.cache_data
def lda_topic_modeling(texts, n_topics=5):
    extra_stopwords = ['said', 'mr', 'just', 'like', 'new', 'time', 'year', 'years']
    cv = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS) + extra_stopwords,
        max_features=5000,
        min_df=0.01,
        token_pattern=r'\b[a-zA-Z]{2,}\b'
    )
    dtm = cv.fit_transform(texts)
    n_jobs = multiprocessing.cpu_count()
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        n_jobs=n_jobs,
        learning_method='batch'
    )
    lda.fit(dtm)

    # Top words
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [cv.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
        topics.append({"LDA Topic": f"Topic {topic_idx}", "Top Keywords": ", ".join(top_words)})

    # Soft assignment for example texts
    lda_probs = lda.transform(dtm)
    return topics, lda_probs

# --- Streamlit App ---
st.set_page_config(page_title="K-Means vs LDA Topic Modeling", page_icon="🧠", layout="wide")
st.title("Topic Modeling Dashboard")
st.write("Upload your dataset to compare embedding-based KMeans and traditional LDA.")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded!")
    st.dataframe(df.head())

    text_col = st.selectbox("Select the text column", df.columns)
    n_rows = st.slider("Number of rows to process", 100, len(df), 500)
    n_clusters = st.slider("Number of topics (clusters)", 2, 10, 5)
    top_n = 50

    if st.button("Run Topic Modeling"):
        with st.spinner("Running topic modeling..."):
            # --- Preprocess ---
            texts, subset_df = preprocess_texts(df, text_col, n_rows)
            st.success("✅ Lemmatization complete!")
            st.dataframe(subset_df[["word_counts"]].head(10))

            # --- KMeans ---
            embeddings = generate_embeddings(texts)
            X_reduced = reduce_dimensions(embeddings)
            clusters = cluster_texts(X_reduced, n_clusters)
            df_result = pd.DataFrame({"Text": texts, "Cluster": clusters})
            st.success("✅ K-Means clustering complete!")

            # KMeans top words
            topics_df = extract_top_words(df_result, n_clusters, top_n, _model=model)

            # --- LDA ---
            lda_topics, lda_probs = lda_topic_modeling(texts, n_topics=n_clusters)

            # Assign meaningful labels to LDA topics
            candidate_labels = [
                "Politics", "Economy", "Sports", "Technology", "Environment",
                "Health", "Education", "Crime", "Entertainment", "World"
            ]
            label_embeddings = model.encode(candidate_labels)

            lda_topic_labels = []
            for topic_idx, topic_info in enumerate(lda_topics):
                top_words_text = topic_info["Top Keywords"]
                topic_embedding = model.encode([top_words_text]).mean(axis=0)
                sims = cosine_similarity([topic_embedding], label_embeddings)[0]
                best_label = candidate_labels[int(np.argmax(sims))]
                lda_topic_labels.append(best_label)

            # Simplified LDA DataFrame
            lda_topics_df = pd.DataFrame({
                "Topic Label": lda_topic_labels,
                "Top Keywords": [topic_info["Top Keywords"] for topic_info in lda_topics]
            })
            st.success("✅ LDA topic modeling complete!")

            # --- Show tables ---
            st.subheader("🟩 K-Means Topics")
            st.dataframe(topics_df)

            st.subheader("🔄 LDA Topics")
            st.dataframe(lda_topics_df)

            # --- Tabs ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([ "📊 Topic Distribution","📈 Silhouette Scores","📝 Example Texts","🔀 Topic Flow","⏱️ Time Comparison"])

            # Tab 1: Topic distribution
            with tab1:
                lda_assignments = lda_probs.argmax(axis=1)
                df_result["LDA Cluster"] = lda_assignments

                kmeans_counts = df_result["Cluster"].value_counts().sort_index()
                lda_counts = df_result["LDA Cluster"].value_counts().sort_index()

                width = 0.35
                fig, ax = plt.subplots()
                ax.bar(range(n_clusters), kmeans_counts, width, label='KMeans', color='#FBECC5')
                ax.bar([i+width for i in range(n_clusters)], lda_counts, width, label='LDA', color='#F8C7C3')
                ax.set_xlabel("Topics")
                ax.set_ylabel("Number of Documents")
                ax.set_title("Document Distribution Across Topics")
                ax.legend()
                st.pyplot(fig)

            # Tab 2: Silhouette scores
            with tab2:
                st.subheader("📈 Silhouette Score Comparison")
                st.markdown("""
                The **Silhouette Score** is a measure of how well each document fits within its assigned topic cluster.
                It ranges from **-1 to +1**:
                - **+1** → Document is well matched to its cluster and poorly matched to neighboring clusters.
                - **0** → Document lies between two clusters (ambiguous topic).
                - **-1** → Document may be assigned to the wrong cluster.

                A **higher Silhouette Score** indicates clearer and more distinct topic groupings.
                """)

                kmeans_sil = silhouette_score(X_reduced, clusters)
                lda_sil = silhouette_score(lda_probs, lda_probs.argmax(axis=1))

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="KMeans Silhouette Score", value=f"{kmeans_sil:.3f}")
                with col2:
                    st.metric(label="LDA Silhouette Score", value=f"{lda_sil:.3f}")

                st.markdown("K-Means often performs better when semantic embeddings are used, "
                            "while LDA relies purely on word frequency, which can affect its separation quality.")
            
            # Tab 3: Example texts
            with tab3:
                st.subheader("📝 Example Texts by KMeans Topic")
                for cluster_id in range(n_clusters):
                    with st.expander(f"KMeans Topic {cluster_id} - {topics_df['Topic Label'][cluster_id]}"):
                        sample_texts = df_result[df_result["Cluster"] == cluster_id]["Text"].head(5).tolist()
                        for t in sample_texts:
                            st.write(f"• {t[:300]}...")

                st.subheader("📝 Example Texts by LDA Topic")
                threshold = 0.2
                for topic_idx in range(n_clusters):
                    with st.expander(f"LDA Topic {topic_idx} - {lda_topic_labels[topic_idx]}"):
                        docs = [texts[i] for i, prob in enumerate(lda_probs[:, topic_idx]) if prob > threshold]
                        for d in docs[:5]:
                            st.write(f"• {d[:300]}...")

            # Tab 4: Sankey diagram
            with tab4:
                st.write("### 🔀 Topic Flow between KMeans and LDA")
                
                # Sort topics alphabetically by label
                kmeans_sorted_labels = sorted(topics_df["Topic Label"])
                lda_sorted_labels = sorted(lda_topics_df["Topic Label"])

                # Node labels
                all_labels = [f"KMeans: {label}" for label in kmeans_sorted_labels] + \
                            [f"LDA: {label}" for label in lda_sorted_labels]

                # Mapping from label -> node index
                kmeans_label_to_idx = {label: i for i, label in enumerate(kmeans_sorted_labels)}
                lda_label_to_idx = {label: i + n_clusters for i, label in enumerate(lda_sorted_labels)}

                # Links
                source, target, value, link_colors = [], [], [], []
                sankey_colors = ['#6B8AD7', '#BE95DC', '#EFA2D2', '#FCB5B5', '#F9C6AB', '#F8E3BC']
                for k_label in kmeans_sorted_labels:
                    for l_label in lda_sorted_labels:
                        k_idx = kmeans_label_to_idx[k_label]
                        l_idx = lda_label_to_idx[l_label]
                        count = np.sum(
                            (df_result["Cluster"] == topics_df.index[topics_df["Topic Label"] == k_label][0]) &
                            (df_result["LDA Cluster"] == lda_topics_df.index[lda_topics_df["Topic Label"] == l_label][0])
                        )
                        # include link even if count is 0
                        source.append(k_idx)
                        target.append(l_idx)
                        value.append(count)
                        link_colors.append(sankey_colors[k_idx % len(sankey_colors)])


                # Sankey diagram
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        label=all_labels,
                        pad=15,
                        thickness=20,
                        color=sankey_colors * 2,  # repeat for KMeans + LDA nodes
                        line=dict(color='black', width=1),
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color=link_colors
                    )
                )])

                fig.update_layout(
                    title_text="Flow of documents between KMeans and LDA topics",
                    font=dict(size=12, color="black")
                )
                
                st.plotly_chart(fig, use_container_width=True)

            # Tab 5: Time Comparison
            with tab5:
                st.subheader("⏱️ Time Taken Comparison")
                st.markdown("""
                This comparison shows the **execution time** taken by each topic modeling approach 
                (K-Means and LDA) for the same number of topics and documents.
                Use this to understand **efficiency trade-offs** between traditional and modern techniques.
                """)

                import time
                start_kmeans = time.time()
                _ = cluster_texts(X_reduced, n_clusters)
                end_kmeans = time.time()

                start_lda = time.time()
                _ = lda_topic_modeling(texts, n_topics=n_clusters)
                end_lda = time.time()

                kmeans_time = end_kmeans - start_kmeans
                lda_time = end_lda - start_lda

                fig, ax = plt.subplots()
                ax.bar(["K-Means", "LDA"], [kmeans_time, lda_time], color=['#A3D9A5', '#F6B8B8'])
                ax.set_ylabel("Time (seconds)")
                ax.set_title("Execution Time Comparison")
                st.pyplot(fig)

                st.write(f"**K-Means Time:** {kmeans_time:.6f} seconds")
                st.write(f"**LDA Time:** {lda_time:.6f} seconds")
                
            st.subheader("📥 Export Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    label="⬇️ Download K-Means Topics",
                    data=topics_df.to_csv(index=False).encode('utf-8'),
                    file_name="kmeans_topics.csv",
                    mime="text/csv"
                )

            with col2:
                st.download_button(
                    label="⬇️ Download LDA Topics",
                    data=lda_topics_df.to_csv(index=False).encode('utf-8'),
                    file_name="lda_topics.csv",
                    mime="text/csv"
                )

            with col3:
                st.download_button(
                    label="⬇️ Download Combined Results",
                    data=df_result.to_csv(index=False).encode('utf-8'),
                    file_name="combined_results.csv",
                    mime="text/csv"
                )
