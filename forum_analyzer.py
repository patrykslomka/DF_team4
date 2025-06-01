import os
import json
import logging
import pandas as pd
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from bertopic import BERTopic
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import gensim


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forum_analyzer")

class ForumAnalyzer:
    def __init__(self, input_file="processed_data.json", output_dir="analysis_results"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.keybert = KeyBERT(model=self.sentence_model)

    def load_data(self) -> Dict[str, List[str]]:
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return {
            'darkweb': df[df['source'] == 'darkweb']['text'].tolist(),
            'reddit': df[df['source'] == 'reddit']['text'].tolist()
        }

    def run_lda(self, texts: List[str], n_topics=8):
        if not texts:
            return None, [], [], None, None
        vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        doc_topics = lda.fit_transform(X).argmax(axis=1)
        topic_words = []
        topic_word_scores = []
        for idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
            top_scores = [topic[i] for i in top_indices]
            topic_words.append(top_words)
            topic_word_scores.append(top_scores)
        # Aggregate word importances across all topics
        word_importance = np.sum(lda.components_, axis=0)
        word_names = vectorizer.get_feature_names_out()
        return lda, topic_words, doc_topics, word_importance, word_names

    def run_nmf(self, texts: List[str], n_topics=8):
        if not texts:
            return None, [], [], None, None
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        nmf = NMF(n_components=n_topics, random_state=42)
        doc_topics = nmf.fit_transform(X).argmax(axis=1)
        topic_words = []
        topic_word_scores = []
        for idx, topic in enumerate(nmf.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
            top_scores = [topic[i] for i in top_indices]
            topic_words.append(top_words)
            topic_word_scores.append(top_scores)
        # Aggregate word importances across all topics
        word_importance = np.sum(nmf.components_, axis=0)
        word_names = vectorizer.get_feature_names_out()
        return nmf, topic_words, doc_topics, word_importance, word_names

    def run_kmeans(self, texts: List[str], n_topics=8):
        if not texts:
            return None, [], [], None, None
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=n_topics, random_state=42)
        doc_topics = kmeans.fit_predict(X)
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        topic_words = []
        topic_word_scores = []
        for i in range(n_topics):
            top_words = [terms[ind] for ind in order_centroids[i, :10]]
            top_scores = [kmeans.cluster_centers_[i][ind] for ind in order_centroids[i, :10]]
            topic_words.append(top_words)
            topic_word_scores.append(top_scores)
        # Aggregate word importances across all clusters
        word_importance = np.sum(kmeans.cluster_centers_, axis=0)
        word_names = terms
        return kmeans, topic_words, doc_topics, word_importance, word_names

    def run_bertopic(self, texts: List[str], n_topics=8):
        if not texts:
            return None, [], [], None, None
        model = BERTopic(embedding_model=self.sentence_model, min_topic_size=5, nr_topics=n_topics)
        topics, _ = model.fit_transform(texts)
        topic_info = model.get_topic_info()
        topic_words = []
        topic_word_scores = []
        all_words = []
        all_scores = []
        for topic_id in topic_info['Topic'].unique():
            if topic_id != -1:
                words_scores = model.get_topic(topic_id)[:10]
                words = [w for w, _ in words_scores]
                scores = [s for _, s in words_scores]
                topic_words.append(words)
                topic_word_scores.append(scores)
                all_words.extend(words)
                all_scores.extend(scores)
        # Aggregate: sum scores for each word across topics
        word_score_dict = {}
        for w, s in zip(all_words, all_scores):
            word_score_dict[w] = word_score_dict.get(w, 0) + s
        word_names = np.array(list(word_score_dict.keys()))
        word_importance = np.array(list(word_score_dict.values()))
        return model, topic_words, topics, word_importance, word_names

    def plot_top_words_bar(self, word_importance, word_names, source, model_name):
        # Get top 10 words by importance
        idx = np.argsort(word_importance)[-10:][::-1]
        top_words = word_names[idx]
        top_scores = word_importance[idx]
        fig = go.Figure([go.Bar(x=top_words, y=top_scores, marker_color='crimson' if source=='darkweb' else 'royalblue')])
        fig.update_layout(title=f"Top 10 Words for {source.capitalize()} ({model_name})", xaxis_title="Word", yaxis_title="Importance", height=400)
        fig.write_html(os.path.join(self.viz_dir, f"bar_topwords_{source}_{model_name}.html"))

    def plot_topic_prevalence(self, doc_topics_darkweb, doc_topics_reddit, n_topics_darkweb, n_topics_reddit, model_name):
        fig = make_subplots(rows=2, cols=1, subplot_titles=[f"Dark Web ({model_name})", f"Reddit ({model_name})"])
        # Dark Web
        if doc_topics_darkweb is not None:
            counts = pd.Series(doc_topics_darkweb).value_counts().sort_index()
            fig.add_trace(go.Bar(x=[f"T{i}" for i in range(n_topics_darkweb)], y=counts.reindex(np.arange(n_topics_darkweb), fill_value=0).values, marker_color='crimson'), row=1, col=1)
        # Reddit
        if doc_topics_reddit is not None:
            counts = pd.Series(doc_topics_reddit).value_counts().sort_index()
            fig.add_trace(go.Bar(x=[f"T{i}" for i in range(n_topics_reddit)], y=counts.reindex(np.arange(n_topics_reddit), fill_value=0).values, marker_color='royalblue'), row=2, col=1)
        fig.update_layout(title=f"Topic Prevalence - {model_name}", height=600)
        fig.write_html(os.path.join(self.viz_dir, f"topic_prevalence_{model_name}.html"))

    def plot_wordcloud(self, topic_words, source, model_name):
        words = [w for topic in topic_words for w in topic]
        freq = pd.Series(words).value_counts().to_dict()
        wc = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(freq)
        plt.figure(figsize=(10,5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"{source.capitalize()} WordCloud ({model_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, f"wordcloud_{source}_{model_name}.png"))
        plt.close()

    def run_analysis(self):
        data = self.load_data()
        n_topics = 8
        for source in ['darkweb', 'reddit']:
            texts = data[source]
            logger.info(f"Running LDA for {source}...")
            _, lda_topics, lda_doc_topics, lda_word_importance, lda_word_names = self.run_lda(texts, n_topics)
            logger.info(f"Running NMF for {source}...")
            _, nmf_topics, nmf_doc_topics, nmf_word_importance, nmf_word_names = self.run_nmf(texts, n_topics)
            logger.info(f"Running KMeans for {source}...")
            _, kmeans_topics, kmeans_doc_topics, kmeans_word_importance, kmeans_word_names = self.run_kmeans(texts, n_topics)
            logger.info(f"Running BERTopic for {source}...")
            _, bertopic_topics, bertopic_doc_topics, bertopic_word_importance, bertopic_word_names = self.run_bertopic(texts, n_topics)
            # Visualizations for each model
            for model, word_importance, word_names in [
                ("LDA", lda_word_importance, lda_word_names),
                ("NMF", nmf_word_importance, nmf_word_names),
                ("KMEANS", kmeans_word_importance, kmeans_word_names),
                ("BERTOPIC", bertopic_word_importance, bertopic_word_names)
            ]:
                self.plot_top_words_bar(word_importance, word_names, source, model)
            # Topic prevalence and wordclouds
            self.plot_topic_prevalence(lda_doc_topics, nmf_doc_topics, n_topics, n_topics, "LDA_NMF")
            self.plot_wordcloud(lda_topics, source, "LDA")
            self.plot_wordcloud(nmf_topics, source, "NMF")
            self.plot_wordcloud(kmeans_topics, source, "KMEANS")
            self.plot_wordcloud(bertopic_topics, source, "BERTOPIC")
        logger.info("Analysis and visualizations complete.")

def main():
    analyzer = ForumAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 