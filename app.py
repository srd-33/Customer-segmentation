import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from model import load_data, kmeans_clustering, som_clustering, elbow_method

st.set_page_config(layout="wide")
st.title("🛍 Customer Segmentation: k-Means vs SOM")

# Sidebar settings
st.sidebar.header("Settings")
n_clusters = st.sidebar.slider("Number of Clusters (k-Means)", 2, 10, 5)
som_x = st.sidebar.slider("SOM Grid X", 2, 10, 5)
som_y = st.sidebar.slider("SOM Grid Y", 2, 10, 5)

# Load data
data = load_data()
st.write("### Sample Data", data.head())

# Run clustering
data_kmeans, kmeans_model, km_score = kmeans_clustering(data, n_clusters)
data_som, som_model, som_score = som_clustering(data, x=som_x, y=som_y)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["⚙️ k-Means", "🧠 SOM", "📊 Comparison", "📈 Elbow Method"])

# --- Tab 1: k-Means ---
with tab1:
    st.subheader("k-Means Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster",
                    data=data_kmeans, palette="Set2", ax=ax)
    plt.title("k-Means Clusters")
    st.pyplot(fig)
    st.write(f"**Silhouette Score (k-Means):** {km_score:.3f}")
    st.write("Cluster Counts:", data_kmeans['Cluster'].value_counts().to_dict())

# --- Tab 2: SOM ---
with tab2:
    st.subheader("Self-Organizing Map (SOM) Clustering")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster",
                    data=data_som, palette="tab10", ax=ax2)
    plt.title("SOM Clusters")
    st.pyplot(fig2)
    if som_score == -1:
        st.write("**Silhouette Score (SOM):** Not valid (only one cluster found)")
    else:
        st.write(f"**Silhouette Score (SOM):** {som_score:.3f}")
    st.write("Cluster Counts:", data_som['Cluster'].value_counts().to_dict())

# --- Tab 3: Comparison ---
with tab3:
    st.subheader("Silhouette Score Comparison")
    methods = ["k-Means", "SOM"]
    scores = [km_score, som_score if som_score != -1 else 0]

    fig3, ax3 = plt.subplots()
    bars = ax3.bar(methods, scores, color=["#4CAF50", "#2196F3"])
    ax3.set_ylabel("Silhouette Score")
    ax3.set_ylim(0, 1)
    ax3.set_title("k-Means vs SOM: Silhouette Score")
    for bar, score in zip(bars, scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    st.pyplot(fig3)

    st.write("### Cluster Distribution Side by Side")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**k-Means**")
        st.dataframe(data_kmeans['Cluster'].value_counts())
    with col2:
        st.write("**SOM**")
        st.dataframe(data_som['Cluster'].value_counts())

    # --- Side-by-side scatter plot using PCA ---
    st.write("### Cluster Visualization (PCA 2D)")
    X = data.values
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    kmeans_labels = data_kmeans['Cluster'].values
    som_labels = data_som['Cluster'].values

    fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="tab10", s=40, alpha=0.8)
    axes[0].set_title("k-Means Clustering (PCA 2D)")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=som_labels, cmap="tab10", s=40, alpha=0.8)
    axes[1].set_title("SOM Clustering (PCA 2D)")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    st.pyplot(fig4)

# --- Tab 4: Elbow Method ---
with tab4:
    st.subheader("Elbow Method for Optimal k")
    max_k = st.slider("Max number of clusters to test", 5, 15, 10)
    K, inertias, optimal_k = elbow_method(data, k_range=max_k)

    fig5, ax5 = plt.subplots()
    ax5.plot(K, inertias, marker="o", linestyle="--")
    ax5.set_xlabel("Number of Clusters (k)")
    ax5.set_ylabel("Inertia (Within-Cluster SSE)")
    ax5.set_title("Elbow Method for k-Means")
    if optimal_k is not None:
        elbow_inertia = inertias[K.index(optimal_k)]
        ax5.scatter(optimal_k, elbow_inertia, s=200, c="red", marker="X", label=f"Elbow: k={optimal_k}")
        ax5.legend()
    st.pyplot(fig5)
    if optimal_k is not None:
        st.success(f"✅ Optimal number of clusters (detected elbow point): **k = {optimal_k}**")
    else:
        st.warning("⚠️ No clear elbow point detected. Try adjusting the max_k range.")
