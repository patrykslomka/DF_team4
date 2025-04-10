"""
SuprBay OpSec Discourse Analyzer
--------------------------------
A specialized tool for analyzing OpSec discussions from SuprBay Dark Web forum
and comparing them with Surface Web discourse.

This script provides:
- Specialized topic modeling for OpSec discussions
- Dark Web specific terminology analysis
- Comparison with Surface Web data
- Visualizations focused on comparative discourse analysis
"""

import os
import glob
import json
import argparse
import logging
from datetime import datetime
from collections import Counter

# Data analysis libraries
import pandas as pd
import numpy as np

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Topic modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("suprbay_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("suprbay_analyzer")

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')


class SuprBayAnalyzer:
    """Analyzer for SuprBay OpSec discourse data"""

    def __init__(self, darkweb_dir="data/darkweb",
                 surfaceweb_dir="data/surfaceweb",
                 reddit_dir="data/reddit",
                 output_dir="analysis_results"):
        """
        Initialize the analyzer

        Args:
            darkweb_dir (str): Directory containing Dark Web data
            surfaceweb_dir (str): Directory containing Surface Web data
            reddit_dir (str): Directory containing Reddit data
            output_dir (str): Directory to store analysis results
        """
        self.darkweb_dir = darkweb_dir
        self.surfaceweb_dir = surfaceweb_dir
        self.reddit_dir = reddit_dir
        self.output_dir = output_dir

        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Define OpSec-specific stopwords
        self.opsec_stopwords = {
            'opsec', 'security', 'privacy', 'http', 'https', 'www', 'com',
            'html', 'org', 'net', 'thread', 'post', 'comment', 'reply',
            'edit', 'edited', 'user', 'users', 'username', 'anonymous',
            'forum', 'forums', 'board', 'boards', 'just', 'like', 'get',
            'got', 'going', 'go', 'use', 'using', 'used', 'can', 'could',
            'would', 'will', 'one', 'two', 'three', 'make', 'want', 'need',
            'know', 'good', 'really', 'thing', 'things', 'think', 'way',
            'getting', 'even', 'suprbay', 'piratebay', 'onion'
        }

        # Add domain-specific stopwords
        self.stop_words.update(self.opsec_stopwords)

        # Create output directory structure
        self.viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        # Set dark theme for visualizations
        plt.style.use('dark_background')

        # Initialize data storage
        self.darkweb_data = None
        self.surfaceweb_data = None
        self.reddit_data = None
        self.combined_surface_data = None

    def load_data(self):
        """
        Load data from JSON files

        Returns:
            dict: Dictionary containing DataFrames with the loaded data
        """
        # Find all JSON files
        darkweb_files = glob.glob(os.path.join(self.darkweb_dir, "*.json"))
        surfaceweb_files = glob.glob(os.path.join(self.surfaceweb_dir, "*.json"))
        reddit_files = glob.glob(os.path.join(self.reddit_dir, "*.json"))

        logger.info(f"Found {len(darkweb_files)} Dark Web data files")
        logger.info(f"Found {len(surfaceweb_files)} Surface Web data files")
        logger.info(f"Found {len(reddit_files)} Reddit data files")

        # Load data
        darkweb_data = []
        surfaceweb_data = []
        reddit_data = []

        # Load Dark Web data
        for file_path in darkweb_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    darkweb_data.extend(data)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")

        # Load Surface Web data
        for file_path in surfaceweb_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    surfaceweb_data.extend(data)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")

        # Load Reddit data
        for file_path in reddit_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    reddit_data.extend(data)
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")

        # Convert to DataFrames
        self.darkweb_data = pd.DataFrame(darkweb_data) if darkweb_data else pd.DataFrame()
        self.surfaceweb_data = pd.DataFrame(surfaceweb_data) if surfaceweb_data else pd.DataFrame()
        self.reddit_data = pd.DataFrame(reddit_data) if reddit_data else pd.DataFrame()

        # Combine surface web and reddit data
        if not self.surfaceweb_data.empty and not self.reddit_data.empty:
            self.combined_surface_data = pd.concat([self.surfaceweb_data, self.reddit_data])
        elif not self.surfaceweb_data.empty:
            self.combined_surface_data = self.surfaceweb_data
        elif not self.reddit_data.empty:
            self.combined_surface_data = self.reddit_data
        else:
            self.combined_surface_data = pd.DataFrame()

        logger.info(f"Loaded {len(self.darkweb_data)} Dark Web posts")
        logger.info(f"Loaded {len(self.surfaceweb_data)} Surface Web posts")
        logger.info(f"Loaded {len(self.reddit_data)} Reddit posts")
        logger.info(f"Combined Surface Web data: {len(self.combined_surface_data)} posts")

        return {
            'darkweb': self.darkweb_data,
            'surfaceweb': self.surfaceweb_data,
            'reddit': self.reddit_data,
            'combined_surface': self.combined_surface_data
        }

    def preprocess_text(self, text):
        """
        Preprocess text for analysis

        Args:
            text (str): Raw text

        Returns:
            list: Preprocessed tokens
        """
        if not isinstance(text, str) or not text:
            return []

        # Tokenize
        tokens = word_tokenize(text.lower())

        # Remove stopwords, non-alphanumeric tokens, and short tokens
        tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words and len(t) > 2]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def clean_corpus(self, df):
        """
        Clean and prepare a corpus from DataFrame

        Args:
            df (pd.DataFrame): DataFrame with content column

        Returns:
            tuple: (clean_df, docs, preprocessed_docs)
        """
        if df.empty:
            return df, [], []

        # Filter out empty content
        clean_df = df[df['content'].notna() & (df['content'] != '')]

        # Extract documents
        docs = clean_df['content'].tolist()

        # Preprocess each document
        preprocessed_docs = [' '.join(self.preprocess_text(doc)) for doc in docs]

        return clean_df, docs, preprocessed_docs

    def extract_keywords(self, texts, n=20):
        """
        Extract top keywords using TF-IDF

        Args:
            texts (list): List of text documents
            n (int): Number of top keywords to extract

        Returns:
            list: Top keywords with their scores
        """
        if not texts:
            return []

        # Create TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        # Transform texts
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        # Get feature names
        feature_names = tfidf_vectorizer.get_feature_names_out()

        # Calculate average TF-IDF score for each term
        avg_tfidf = np.mean(tfidf_matrix.toarray(), axis=0)

        # Create a list of (term, score) tuples
        keywords = [(feature_names[i], avg_tfidf[i]) for i in range(len(feature_names))]

        # Sort by score and take top n
        keywords.sort(key=lambda x: x[1], reverse=True)
        return keywords[:n]

    def extract_opsec_specific_terminology(self, source_texts, reference_texts, n=20):
        """
        Extract OpSec-specific terminology by comparing with a reference corpus

        Args:
            source_texts (list): OpSec-related texts
            reference_texts (list): General reference texts
            n (int): Number of top terms to extract

        Returns:
            list: OpSec-specific terms with their distinctiveness scores
        """
        if not source_texts or not reference_texts:
            return []

        # Vectorize both corpora
        count_vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        # Fit the vectorizer on all texts
        all_texts = source_texts + reference_texts
        count_vectorizer.fit(all_texts)

        # Transform both corpora
        source_matrix = count_vectorizer.transform(source_texts)
        reference_matrix = count_vectorizer.transform(reference_texts)

        # Calculate relative frequencies
        source_freq = np.sum(source_matrix.toarray(), axis=0) / len(source_texts)
        reference_freq = np.sum(reference_matrix.toarray(), axis=0) / len(reference_texts)

        # Avoid division by zero
        reference_freq = np.where(reference_freq == 0, 0.1, reference_freq)

        # Calculate ratio of frequencies (distinctiveness score)
        distinctiveness = source_freq / reference_freq

        # Get feature names
        feature_names = count_vectorizer.get_feature_names_out()

        # Create a list of (term, score) tuples
        domain_terms = [(feature_names[i], distinctiveness[i]) for i in range(len(feature_names))]

        # Sort by score and take top n
        domain_terms.sort(key=lambda x: x[1], reverse=True)
        return domain_terms[:n]

    def perform_topic_modeling(self, texts, n_topics=5, method='lda'):
        """
        Perform topic modeling on texts

        Args:
            texts (list): List of text documents
            n_topics (int): Number of topics to extract
            method (str): Method to use ('lda' or 'nmf')

        Returns:
            tuple: (model, vectorizer, transformed_data, feature_names)
        """
        if not texts:
            return None, None, None, None

        if method.lower() == 'lda':
            # LDA works better with raw term counts
            count_vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            count_matrix = count_vectorizer.fit_transform(texts)

            # Create and fit LDA model
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20
            )
            transformed = lda_model.fit_transform(count_matrix)

            return lda_model, count_vectorizer, transformed, count_vectorizer.get_feature_names_out()

        else:  # NMF
            # Create TF-IDF vectorizer
            tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            # Transform texts
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

            # Create and fit NMF model
            nmf_model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=1000
            )
            transformed = nmf_model.fit_transform(tfidf_matrix)

            return nmf_model, tfidf_vectorizer, transformed, tfidf_vectorizer.get_feature_names_out()

    def get_topic_keywords(self, model, feature_names, n_keywords=10):
        """
        Get top keywords for each topic

        Args:
            model: Fitted topic model (LDA or NMF)
            feature_names: Feature names from vectorizer
            n_keywords (int): Number of keywords per topic

        Returns:
            list: List of topic keywords
        """
        if model is None:
            return []

        topic_keywords = []

        for topic_idx, topic in enumerate(model.components_):
            # Get top keywords for this topic
            top_keywords_idx = topic.argsort()[:-n_keywords-1:-1]
            top_keywords = [feature_names[i] for i in top_keywords_idx]
            topic_keywords.append(top_keywords)

        return topic_keywords

    def analyze_linguistic_complexity(self, texts):
        """
        Analyze linguistic complexity of texts

        Args:
            texts (list): List of text documents

        Returns:
            dict: Complexity metrics
        """
        if not texts:
            return {}

        # Initialize metrics
        word_counts = []
        sentence_counts = []
        word_lengths = []
        lexical_diversity = []

        for text in texts:
            if not isinstance(text, str) or not text:
                continue

            # Tokenize
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())

            # Filter out non-alphanumeric tokens
            words = [w for w in words if w.isalnum()]

            if not words:
                continue

            # Calculate metrics
            word_counts.append(len(words))
            sentence_counts.append(len(sentences))
            word_lengths.extend([len(word) for word in words])

            # Lexical diversity (unique words / total words)
            unique_words = len(set(words))
            total_words = len(words)
            diversity = unique_words / total_words if total_words > 0 else 0
            lexical_diversity.append(diversity)

        # Aggregate metrics
        return {
            'avg_words_per_text': np.mean(word_counts) if word_counts else 0,
            'avg_sentences_per_text': np.mean(sentence_counts) if sentence_counts else 0,
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
            'avg_lexical_diversity': np.mean(lexical_diversity) if lexical_diversity else 0,
            'median_words_per_text': np.median(word_counts) if word_counts else 0,
            'median_sentences_per_text': np.median(sentence_counts) if sentence_counts else 0
        }

    def perform_sentiment_analysis(self, texts):
        """
        Perform sentiment analysis on texts

        Args:
            texts (list): List of text documents

        Returns:
            pd.DataFrame: DataFrame with sentiment scores
        """
        if not texts:
            return pd.DataFrame()

        # Initialize VADER sentiment analyzer
        sia = SentimentIntensityAnalyzer()

        # Analyze sentiment for each text
        sentiments = []
        for text in texts:
            if isinstance(text, str) and text:
                sentiment = sia.polarity_scores(text)
                sentiments.append(sentiment)
            else:
                sentiments.append({'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0})

        # Convert to DataFrame
        sentiment_df = pd.DataFrame(sentiments)

        return sentiment_df

    def analyze_security_focus(self, docs, keywords):
        """
        Analyze security focus using predefined security-related keywords

        Args:
            docs (list): List of document texts
            keywords (list): List of security-related keywords

        Returns:
            dict: Security focus metrics
        """
        if not docs:
            return {}

        # Initialize counters
        security_mention_count = 0
        docs_with_security_mentions = 0
        keyword_counts = Counter()

        for doc in docs:
            if not isinstance(doc, str) or not doc:
                continue

            doc_lower = doc.lower()

            # Check for security keyword mentions
            doc_mentions = 0
            for keyword in keywords:
                count = doc_lower.count(keyword.lower())
                if count > 0:
                    keyword_counts[keyword] += count
                    doc_mentions += count

            # Update counters
            security_mention_count += doc_mentions
            if doc_mentions > 0:
                docs_with_security_mentions += 1

        # Calculate metrics
        total_docs = len(docs)
        docs_with_mentions_pct = (docs_with_security_mentions / total_docs * 100) if total_docs > 0 else 0
        avg_mentions_per_doc = security_mention_count / total_docs if total_docs > 0 else 0

        # Get top mentioned keywords
        top_keywords = keyword_counts.most_common(10)

        return {
            'total_security_mentions': security_mention_count,
            'docs_with_security_mentions': docs_with_security_mentions,
            'docs_with_mentions_pct': docs_with_mentions_pct,
            'avg_mentions_per_doc': avg_mentions_per_doc,
            'top_security_keywords': top_keywords
        }

    def perform_comparative_analysis(self):
        """
        Perform comprehensive comparative analysis

        Returns:
            dict: Analysis results
        """
        # Load data if not already loaded
        if self.darkweb_data is None or self.combined_surface_data is None:
            self.load_data()

        # Check if we have data to analyze
        if self.darkweb_data.empty and self.combined_surface_data.empty:
            logger.error("No data available for analysis")
            return {}

        # Clean and prepare data
        darkweb_clean, darkweb_docs, darkweb_preprocessed = self.clean_corpus(self.darkweb_data)
        surface_clean, surface_docs, surface_preprocessed = self.clean_corpus(self.combined_surface_data)

        logger.info("Performing comparative analysis...")
        results = {
            'darkweb': {},
            'surface': {},
            'comparison': {}
        }

        # 1. Basic text statistics
        logger.info("Analyzing text statistics...")
        results['darkweb']['text_stats'] = self.analyze_linguistic_complexity(darkweb_docs)
        results['surface']['text_stats'] = self.analyze_linguistic_complexity(surface_docs)

        # 2. Keyword extraction
        logger.info("Extracting keywords...")
        results['darkweb']['top_keywords'] = self.extract_keywords(darkweb_docs)
        results['surface']['top_keywords'] = self.extract_keywords(surface_docs)

        # 3. OpSec-specific terminology
        logger.info("Identifying OpSec-specific terminology...")
        results['darkweb']['opsec_terms'] = self.extract_opsec_specific_terminology(
            darkweb_docs, surface_docs)
        results['surface']['opsec_terms'] = self.extract_opsec_specific_terminology(
            surface_docs, darkweb_docs)

        # 4. Topic modeling
        logger.info("Performing topic modeling...")
        darkweb_model, darkweb_vectorizer, darkweb_topics, darkweb_features = self.perform_topic_modeling(
            darkweb_docs)
        surface_model, surface_vectorizer, surface_topics, surface_features = self.perform_topic_modeling(
            surface_docs)

        results['darkweb']['topic_keywords'] = self.get_topic_keywords(darkweb_model, darkweb_features)
        results['surface']['topic_keywords'] = self.get_topic_keywords(surface_model, surface_features)

        # 5. Sentiment analysis
        logger.info("Analyzing sentiment...")
        darkweb_sentiment = self.perform_sentiment_analysis(darkweb_docs)
        surface_sentiment = self.perform_sentiment_analysis(surface_docs)

        if not darkweb_sentiment.empty:
            results['darkweb']['sentiment'] = {
                'mean': darkweb_sentiment.mean().to_dict(),
                'median': darkweb_sentiment.median().to_dict()
            }
        else:
            results['darkweb']['sentiment'] = {}

        if not surface_sentiment.empty:
            results['surface']['sentiment'] = {
                'mean': surface_sentiment.mean().to_dict(),
                'median': surface_sentiment.median().to_dict()
            }
        else:
            results['surface']['sentiment'] = {}

        # 6. Security focus analysis
        logger.info("Analyzing security focus...")

        # Define security-related keywords for OpSec
        security_keywords = [
            'encryption', 'vpn', 'tor', 'proxy', 'anonymous', 'privacy',
            'security', 'tracking', 'surveillance', 'monitor', 'protect',
            'secure', 'cipher', 'private', 'trace', 'identity', 'hide',
            'mask', 'conceal', 'data', 'information', 'breach', 'leak',
            'vulnerability', 'exploit', 'attack', 'threat', 'risk',
            'exposure', 'compromise', 'hacker', 'adversary', 'agency',
            'government', 'law enforcement', 'backdoor', 'fingerprint'
        ]

        results['darkweb']['security_focus'] = self.analyze_security_focus(darkweb_docs, security_keywords)
        results['surface']['security_focus'] = self.analyze_security_focus(surface_docs, security_keywords)

        # 7. Comparative metrics
        logger.info("Computing comparative metrics...")

        # Find common and unique keywords
        darkweb_keywords = [kw[0] for kw in results['darkweb']['top_keywords']]
        surface_keywords = [kw[0] for kw in results['surface']['top_keywords']]

        common_keywords = set(darkweb_keywords).intersection(set(surface_keywords))
        unique_darkweb_keywords = set(darkweb_keywords).difference(set(surface_keywords))
        unique_surface_keywords = set(surface_keywords).difference(set(darkweb_keywords))

        results['comparison']['common_keywords'] = list(common_keywords)
        results['comparison']['unique_darkweb_keywords'] = list(unique_darkweb_keywords)
        results['comparison']['unique_surface_keywords'] = list(unique_surface_keywords)

        # Compare sentiment
        if results['darkweb']['sentiment'] and results['surface']['sentiment']:
            results['comparison']['sentiment_diff'] = {
                k: results['darkweb']['sentiment']['mean'][k] - results['surface']['sentiment']['mean'][k]
                for k in results['darkweb']['sentiment']['mean'].keys()
            }

        # Compare text complexity
        if results['darkweb']['text_stats'] and results['surface']['text_stats']:
            results['comparison']['complexity_diff'] = {
                k: results['darkweb']['text_stats'][k] - results['surface']['text_stats'][k]
                for k in results['darkweb']['text_stats'].keys()
                if k in results['surface']['text_stats']
            }

        # Compare security focus
        if results['darkweb']['security_focus'] and results['surface']['security_focus']:
            results['comparison']['security_focus_diff'] = {
                'avg_mentions_diff': (results['darkweb']['security_focus']['avg_mentions_per_doc'] -
                                     results['surface']['security_focus']['avg_mentions_per_doc']),
                'docs_with_mentions_pct_diff': (results['darkweb']['security_focus']['docs_with_mentions_pct'] -
                                               results['surface']['security_focus']['docs_with_mentions_pct'])
            }

        # 8. Generate visualizations
        logger.info("Generating visualizations...")
        self.generate_visualizations(results, darkweb_clean, surface_clean)

        # Save the results
        self.save_results(results)

        return results

    def save_results(self, results):
        """
        Save analysis results to a JSON file

        Args:
            results (dict): Analysis results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.output_dir, f"opsec_analysis_results_{timestamp}.json")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        logger.info(f"Analysis results saved to {filepath}")

    def generate_visualizations(self, results, darkweb_df, surface_df):
        """
        Generate visualizations for the analysis results

        Args:
            results (dict): Analysis results
            darkweb_df (pd.DataFrame): Dark Web data
            surface_df (pd.DataFrame): Surface Web data
        """
        # 1. Topic Distribution Comparison
        self.generate_topic_comparison(
            results['darkweb']['topic_keywords'],
            results['surface']['topic_keywords'],
            os.path.join(self.viz_dir, "topic_comparison.png")
        )

        # 2. Security Focus Comparison
        if 'security_focus' in results['darkweb'] and 'security_focus' in results['surface']:
            self.generate_security_focus_comparison(
                results['darkweb']['security_focus'],
                results['surface']['security_focus'],
                os.path.join(self.viz_dir, "security_focus_comparison.png")
            )

        # 3. Sentiment Comparison
        if 'sentiment' in results['darkweb'] and 'sentiment' in results['surface']:
            self.generate_sentiment_comparison(
                results['darkweb']['sentiment'],
                results['surface']['sentiment'],
                os.path.join(self.viz_dir, "sentiment_comparison.png")
            )

        # 4. Text Complexity Comparison
        if 'text_stats' in results['darkweb'] and 'text_stats' in results['surface']:
            self.generate_complexity_comparison(
                results['darkweb']['text_stats'],
                results['surface']['text_stats'],
                os.path.join(self.viz_dir, "complexity_comparison.png")
            )

        # 5. OpSec-Specific Terminology Comparison
        if 'opsec_terms' in results['darkweb'] and 'opsec_terms' in results['surface']:
            self.generate_opsec_terminology_comparison(
                results['darkweb']['opsec_terms'],
                results['surface']['opsec_terms'],
                os.path.join(self.viz_dir, "opsec_terminology_comparison.png")
            )

        # 6. Thread Topics Word Cloud
        if not darkweb_df.empty and 'thread_title' in darkweb_df.columns:
            self.generate_thread_topics_wordcloud(
                darkweb_df['thread_title'].tolist(),
                os.path.join(self.viz_dir, "darkweb_thread_topics.png")
            )

        if not surface_df.empty and 'thread_title' in surface_df.columns:
            self.generate_thread_topics_wordcloud(
                surface_df['thread_title'].tolist(),
                os.path.join(self.viz_dir, "surface_thread_topics.png"),
                is_darkweb=False
            )

    def generate_topic_comparison(self, darkweb_topics, surface_topics, output_path):
        """
        Generate topic comparison visualization

        Args:
            darkweb_topics (list): Dark Web topic keywords
            surface_topics (list): Surface Web topic keywords
            output_path (str): Path to save the visualization
        """
        if not darkweb_topics or not surface_topics:
            logger.warning("Not enough data for topic comparison visualization")
            return

        # Configure the visualization
        num_topics = min(len(darkweb_topics), len(surface_topics), 5)  # Compare up to 5 topics

        # Set up the figure
        fig, axes = plt.subplots(2, num_topics, figsize=(5*num_topics, 10))
        fig.suptitle('Dark Web vs. Surface Web OpSec Topics', fontsize=22, y=0.98)

        # Define color maps
        darkweb_cmap = plt.cm.Reds
        surface_cmap = plt.cm.Blues

        # Plot Dark Web topics
        for i in range(num_topics):
            ax = axes[0, i] if num_topics > 1 else axes[0]
            topic_dict = {word: 1 for word in darkweb_topics[i][:15]}
            wc = WordCloud(
                width=400, height=300,
                background_color='black',
                colormap=darkweb_cmap,
                max_words=50
            ).generate_from_frequencies(topic_dict)

            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Dark Web Topic {i+1}', color='white', fontsize=16)

        # Plot Surface Web topics
        for i in range(num_topics):
            ax = axes[1, i] if num_topics > 1 else axes[1]
            topic_dict = {word: 1 for word in surface_topics[i][:15]}
            wc = WordCloud(
                width=400, height=300,
                background_color='black',
                colormap=surface_cmap,
                max_words=50
            ).generate_from_frequencies(topic_dict)

            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Surface Web Topic {i+1}', color='white', fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Topic comparison visualization saved to {output_path}")

    def generate_security_focus_comparison(self, darkweb_security, surface_security, output_path):
        """
        Generate security focus comparison visualization

        Args:
            darkweb_security (dict): Dark Web security focus metrics
            surface_security (dict): Surface Web security focus metrics
            output_path (str): Path to save the visualization
        """
        if not darkweb_security or not surface_security:
            logger.warning("Not enough data for security focus comparison visualization")
            return

        plt.figure(figsize=(14, 10))

        # Create subplots
        gs = plt.GridSpec(2, 2, height_ratios=[1, 1.5])

        # 1. Security mention metrics
        ax1 = plt.subplot(gs[0, :])

        metrics = ['avg_mentions_per_doc', 'docs_with_mentions_pct']
        labels = ['Avg. Security Mentions per Post', '% Posts with Security Mentions']

        darkweb_values = [darkweb_security.get(m, 0) for m in metrics]
        surface_values = [surface_security.get(m, 0) for m in metrics]

        x = np.arange(len(labels))
        width = 0.35

        ax1.bar(x - width/2, darkweb_values, width, label='Dark Web', color='crimson')
        ax1.bar(x + width/2, surface_values, width, label='Surface Web', color='royalblue')

        ax1.set_ylabel('Value')
        ax1.set_title('Security Focus Metrics', fontsize=18)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()

        # Add values on top of bars
        for i, v in enumerate(darkweb_values):
            ax1.text(i - width/2, v + 0.1, f'{v:.2f}', ha='center', va='bottom', color='white')

        for i, v in enumerate(surface_values):
            ax1.text(i + width/2, v + 0.1, f'{v:.2f}', ha='center', va='bottom', color='white')

        # 2. Top security keywords - Dark Web
        ax2 = plt.subplot(gs[1, 0])

        if 'top_security_keywords' in darkweb_security and darkweb_security['top_security_keywords']:
            kw_dark = darkweb_security['top_security_keywords'][:8]  # Top 8 keywords
            keywords_dark = [item[0] for item in kw_dark]
            counts_dark = [item[1] for item in kw_dark]

            ax2.barh(keywords_dark, counts_dark, color='crimson')
            ax2.set_title('Top Dark Web Security Keywords', fontsize=16)
            ax2.set_xlabel('Frequency')

            # Add values at the end of bars
            for i, v in enumerate(counts_dark):
                ax2.text(v + 0.5, i, str(v), va='center', color='white')

        # 3. Top security keywords - Surface Web
        ax3 = plt.subplot(gs[1, 1])

        if 'top_security_keywords' in surface_security and surface_security['top_security_keywords']:
            kw_surface = surface_security['top_security_keywords'][:8]  # Top 8 keywords
            keywords_surface = [item[0] for item in kw_surface]
            counts_surface = [item[1] for item in kw_surface]

            ax3.barh(keywords_surface, counts_surface, color='royalblue')
            ax3.set_title('Top Surface Web Security Keywords', fontsize=16)
            ax3.set_xlabel('Frequency')

            # Add values at the end of bars
            for i, v in enumerate(counts_surface):
                ax3.text(v + 0.5, i, str(v), va='center', color='white')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Security focus comparison visualization saved to {output_path}")

    def generate_sentiment_comparison(self, darkweb_sentiment, surface_sentiment, output_path):
        """
        Generate sentiment comparison visualization

        Args:
            darkweb_sentiment (dict): Dark Web sentiment metrics
            surface_sentiment (dict): Surface Web sentiment metrics
            output_path (str): Path to save the visualization
        """
        if not darkweb_sentiment or not surface_sentiment:
            logger.warning("Not enough data for sentiment comparison visualization")
            return

        plt.figure(figsize=(12, 8))

        # Prepare data
        categories = ['Negative', 'Neutral', 'Positive', 'Compound']
        keys = ['neg', 'neu', 'pos', 'compound']

        if 'mean' in darkweb_sentiment and 'mean' in surface_sentiment:
            dark_values = [darkweb_sentiment['mean'].get(k, 0) for k in keys]
            surface_values = [surface_sentiment['mean'].get(k, 0) for k in keys]

            x = np.arange(len(categories))
            width = 0.35

            # Plot bars
            plt.bar(x - width/2, dark_values, width, label='Dark Web', color='crimson')
            plt.bar(x + width/2, surface_values, width, label='Surface Web', color='royalblue')

            # Customize plot
            plt.ylabel('Score')
            plt.title('Sentiment Analysis Comparison', fontsize=18)
            plt.xticks(x, categories)
            plt.legend()

            # Add values on top of bars
            for i, v in enumerate(dark_values):
                plt.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', color='white')

            for i, v in enumerate(surface_values):
                plt.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center', va='bottom', color='white')

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Sentiment comparison visualization saved to {output_path}")

    def generate_complexity_comparison(self, darkweb_stats, surface_stats, output_path):
        """
        Generate text complexity comparison visualization

        Args:
            darkweb_stats (dict): Dark Web text statistics
            surface_stats (dict): Surface Web text statistics
            output_path (str): Path to save the visualization
        """
        if not darkweb_stats or not surface_stats:
            logger.warning("Not enough data for complexity comparison visualization")
            return

        plt.figure(figsize=(14, 10))

        # Select metrics to visualize
        metrics = [
            'avg_words_per_text',
            'avg_sentences_per_text',
            'avg_word_length',
            'avg_lexical_diversity'
        ]

        labels = [
            'Avg. Words per Post',
            'Avg. Sentences per Post',
            'Avg. Word Length',
            'Lexical Diversity'
        ]

        # Get values
        darkweb_values = [darkweb_stats.get(m, 0) for m in metrics]
        surface_values = [surface_stats.get(m, 0) for m in metrics]

        # Set up bars
        x = np.arange(len(labels))
        width = 0.35

        # Create plot
        plt.bar(x - width/2, darkweb_values, width, label='Dark Web', color='crimson')
        plt.bar(x + width/2, surface_values, width, label='Surface Web', color='royalblue')

        # Customize plot
        plt.ylabel('Value')
        plt.title('Text Complexity Comparison', fontsize=18)
        plt.xticks(x, labels)
        plt.legend()

        # Add values on top of bars
        for i, v in enumerate(darkweb_values):
            plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom', color='white')

        for i, v in enumerate(surface_values):
            plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', va='bottom', color='white')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Complexity comparison visualization saved to {output_path}")

    def generate_opsec_terminology_comparison(self, darkweb_terms, surface_terms, output_path):
        """
        Generate OpSec terminology comparison visualization

        Args:
            darkweb_terms (list): Dark Web-specific OpSec terms
            surface_terms (list): Surface Web-specific OpSec terms
            output_path (str): Path to save the visualization
        """
        if not darkweb_terms or not surface_terms:
            logger.warning("Not enough data for OpSec terminology comparison visualization")
            return

        plt.figure(figsize=(15, 8))

        # Prepare data
        num_terms = 10  # Number of terms to display

        # Extract terms and scores
        dark_terms = [t[0] for t in darkweb_terms[:num_terms]]
        dark_scores = [t[1] for t in darkweb_terms[:num_terms]]

        surface_terms = [t[0] for t in surface_terms[:num_terms]]
        surface_scores = [t[1] for t in surface_terms[:num_terms]]

        # Create subplots
        plt.subplot(1, 2, 1)
        y_pos = np.arange(len(dark_terms))
        plt.barh(y_pos, dark_scores, color='crimson')
        plt.yticks(y_pos, dark_terms)
        plt.title('Dark Web-Specific OpSec Terms', fontsize=16)
        plt.xlabel('Distinctiveness Score')

        plt.subplot(1, 2, 2)
        y_pos = np.arange(len(surface_terms))
        plt.barh(y_pos, surface_scores, color='royalblue')
        plt.yticks(y_pos, surface_terms)
        plt.title('Surface Web-Specific OpSec Terms', fontsize=16)
        plt.xlabel('Distinctiveness Score')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"OpSec terminology comparison visualization saved to {output_path}")

    def generate_thread_topics_wordcloud(self, thread_titles, output_path, is_darkweb=True):
        """
        Generate word cloud visualization of thread topics

        Args:
            thread_titles (list): List of thread titles
            output_path (str): Path to save the visualization
            is_darkweb (bool): Whether the data is from Dark Web
        """
        if not thread_titles:
            logger.warning("Not enough data for thread topics word cloud visualization")
            return

        # Combine all titles
        text = ' '.join(thread_titles)

        # Tokenize and remove stopwords
        tokens = self.preprocess_text(text)
        processed_text = ' '.join(tokens)

        # Create word cloud
        if is_darkweb:
            wc = WordCloud(
                width=800, height=400,
                background_color='black',
                colormap='Reds',
                max_words=100
            ).generate(processed_text)
            title = 'Dark Web OpSec Discussion Topics'
        else:
            wc = WordCloud(
                width=800, height=400,
                background_color='black',
                colormap='Blues',
                max_words=100
            ).generate(processed_text)
            title = 'Surface Web OpSec Discussion Topics'

        # Display word cloud
        plt.figure(figsize=(12, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Thread topics word cloud visualization saved to {output_path}")

    def generate_summary_report(self, results):
        """
        Generate a summary report of the analysis results

        Args:
            results (dict): Analysis results

        Returns:
            str: Markdown-formatted summary report
        """
        if not results:
            return "No analysis results available."

        # Create report header
        report = """
        # OpSec Discourse Analysis: Dark Web vs. Surface Web
        
        ## Executive Summary
        
        This report presents a comparative analysis of Operational Security (OpSec) discussions
        on Dark Web forums versus Surface Web communities. The analysis reveals distinct patterns
        in how security practices, tools, and advice are discussed across these environments.
        
        """

        # Add key findings
        report += "## Key Findings\n\n"

        # Topic differences
        if ('darkweb' in results and 'topic_keywords' in results['darkweb'] and
            'surface' in results and 'topic_keywords' in results['surface']):
            report += "### Topic Differences\n\n"

            report += "**Dark Web Primary Topics:**\n"
            for i, topic in enumerate(results['darkweb']['topic_keywords'][:3]):
                report += f"- Topic {i+1}: {', '.join(topic[:5])}\n"

            report += "\n**Surface Web Primary Topics:**\n"
            for i, topic in enumerate(results['surface']['topic_keywords'][:3]):
                report += f"- Topic {i+1}: {', '.join(topic[:5])}\n"

            report += "\n"

        # Sentiment differences
        if ('comparison' in results and 'sentiment_diff' in results['comparison']):
            report += "### Sentiment Analysis\n\n"

            # Extract the compound sentiment difference
            compound_diff = results['comparison']['sentiment_diff'].get('compound', 0)

            if abs(compound_diff) < 0.05:
                sentiment_finding = "Similar overall sentiment between Dark Web and Surface Web discussions."
            elif compound_diff > 0:
                sentiment_finding = "Dark Web discussions show more positive sentiment than Surface Web discussions."
            else:
                sentiment_finding = "Surface Web discussions show more positive sentiment than Dark Web discussions."

            report += f"{sentiment_finding}\n\n"

        # Security focus differences
        if ('comparison' in results and 'security_focus_diff' in results['comparison']):
            report += "### Security Focus\n\n"

            avg_mentions_diff = results['comparison']['security_focus_diff'].get('avg_mentions_diff', 0)
            pct_diff = results['comparison']['security_focus_diff'].get('docs_with_mentions_pct_diff', 0)

            if avg_mentions_diff > 0:
                security_finding = "Dark Web discussions contain more security-related terminology and focus more explicitly on OpSec concepts."
            else:
                security_finding = "Surface Web discussions contain more security-related terminology, though they may approach the topics differently."

            report += f"{security_finding}\n\n"

        # Distinct terminology
        if ('comparison' in results and
            'unique_darkweb_keywords' in results['comparison'] and
            'unique_surface_keywords' in results['comparison']):

            report += "### Distinct Terminology\n\n"

            report += "**Dark Web-Specific Terms:**\n"
            dark_terms = results['comparison']['unique_darkweb_keywords'][:7]
            report += f"{', '.join(dark_terms)}\n\n"

            report += "**Surface Web-Specific Terms:**\n"
            surface_terms = results['comparison']['unique_surface_keywords'][:7]
            report += f"{', '.join(surface_terms)}\n\n"

        # Linguistic complexity differences
        if ('comparison' in results and 'complexity_diff' in results['comparison']):
            report += "### Linguistic Patterns\n\n"

            # Get the lexical diversity difference
            diversity_diff = results['comparison']['complexity_diff'].get('avg_lexical_diversity', 0)
            words_per_post_diff = results['comparison']['complexity_diff'].get('avg_words_per_text', 0)

            if diversity_diff > 0:
                diversity_finding = "Dark Web discussions show greater linguistic diversity, suggesting more specialized or technical discourse."
            else:
                diversity_finding = "Surface Web discussions show greater linguistic diversity, possibly indicating more varied participants or topics."

            if words_per_post_diff > 0:
                length_finding = "Dark Web posts tend to be longer on average."
            else:
                length_finding = "Surface Web posts tend to be longer on average."

            report += f"{diversity_finding} {length_finding}\n\n"

        # Implications section
        report += """
        ## Implications
        
        The differences in OpSec discourse between Dark Web and Surface Web environments reflect distinct security cultures and threat models. Dark Web participants appear to approach security with different assumptions and priorities compared to Surface Web communities, likely influenced by the perceived higher stakes and different adversarial models.
        
        These findings suggest that comprehensive OpSec education should account for these different discourse patterns and security cultures, addressing both the technical details emphasized in one environment and the practical considerations highlighted in the other.
        """

        # Format the report for better rendering
        report = "\n".join([line.strip() for line in report.split("\n")])

        return report


def main():
    """Command line interface for the analyzer"""
    parser = argparse.ArgumentParser(description="SuprBay OpSec Discourse Analyzer")

    parser.add_argument("--darkweb-dir", default="data/darkweb", help="Directory containing Dark Web data")
    parser.add_argument("--surfaceweb-dir", default="data/surfaceweb", help="Directory containing Surface Web data")
    parser.add_argument("--reddit-dir", default="data/reddit", help="Directory containing Reddit data")
    parser.add_argument("--output-dir", default="analysis_results", help="Directory to store analysis results")
    parser.add_argument("--report", default="opsec_comparative_analysis.md", help="Output file for summary report")

    args = parser.parse_args()

    # Create analyzer
    analyzer = SuprBayAnalyzer(
        darkweb_dir=args.darkweb_dir,
        surfaceweb_dir=args.surfaceweb_dir,
        reddit_dir=args.reddit_dir,
        output_dir=args.output_dir
    )

    # Load data
    logger.info("Loading data...")
    data = analyzer.load_data()

    # Check if we have enough data
    if data['darkweb'].empty and data['combined_surface'].empty:
        logger.error("No data found for analysis. Please ensure your data directories are correct.")
        return 1

    # Perform analysis
    logger.info("Performing comparative analysis...")
    results = analyzer.perform_comparative_analysis()

    # Generate summary report
    logger.info("Generating summary report...")
    report = analyzer.generate_summary_report(results)

    # Save report
    report_path = os.path.join(args.output_dir, args.report)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"Summary report saved to {report_path}")
    logger.info("Analysis complete!")

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)