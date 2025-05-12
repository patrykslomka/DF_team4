"""
Fixed Tomotopy Implementation for Topic Modeling
-----------------------------------------------
Complete implementation with proper document handling in Tomotopy
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

# Standard NLP libraries
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

# Import Tomotopy for sophisticated topic modeling
import tomotopy as tp

# Import BERTopic
try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("BERTopic not available. Install with 'pip install bertopic'")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("topic_modeling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("topic_modeling")


class AdvancedTopicModeler:
    """Advanced topic modeling with multiple algorithms and evaluation metrics"""
    
    def __init__(self, output_dir="analysis_results/topic_models"):
        """
        Initialize the topic modeler
        
        Args:
            output_dir (str): Directory to store topic modeling results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dictionary to store models
        self.models = {}
        self.model_results = {}
        self.evaluation = {}
        self.topic_comparison = None
        
    def preprocess_texts(self, texts, min_token_length=3):
        """
        Preprocess texts for topic modeling
        
        Args:
            texts (list): List of text documents
            min_token_length (int): Minimum token length
            
        Returns:
            tuple: (processed_docs, dictionary, corpus)
        """
        logger.info("Preprocessing texts for topic modeling...")
        
        # Tokenize texts
        tokenized_docs = []
        for text in texts:
            if not isinstance(text, str) or not text:
                continue
            
            # Tokenize and clean
            tokens = word_tokenize(text.lower())
            # Filter tokens by length and alphanumeric
            tokens = [token for token in tokens if token.isalnum() and len(token) >= min_token_length]
            tokenized_docs.append(tokens)
        
        # Create dictionary and corpus for Gensim
        dictionary = Dictionary(tokenized_docs)
        # Filter extremes (rare and common words)
        dictionary.filter_extremes(no_below=5, no_above=0.7)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        return tokenized_docs, dictionary, corpus
    
    def train_lda_tomotopy(self, texts, num_topics=10, iterations=1000):
        """
        Train LDA model using Tomotopy - Fixed version
        
        Args:
            texts (list): List of text documents
            num_topics (int): Number of topics
            iterations (int): Number of iterations
            
        Returns:
            tp.LDAModel: Trained LDA model
        """
        logger.info(f"Training LDA model with Tomotopy (k={num_topics})...")
        
        # Create LDA model
        lda_model = tp.LDAModel(k=num_topics)
        
        # Process and add documents properly
        for doc in texts:
            if isinstance(doc, str) and doc:
                # Split text into words
                words = [word for word in doc.lower().split() if len(word) >= 3]
                # Add document to the model
                if words:
                    lda_model.add_doc(words)
        
        # Train the model
        logger.info(f"Training LDA model for {iterations} iterations...")
        for i in tqdm(range(0, iterations, 10)):
            lda_model.train(10)
        
        # Store the model
        self.models['lda'] = lda_model
        
        # Get model results
        result = self.extract_tomotopy_results(lda_model)
        self.model_results['lda'] = result
        
        return lda_model
    
    def train_ctm_tomotopy(self, texts, num_topics=10, iterations=1000):
        """
        Train CTM (Correlated Topic Model) using Tomotopy - Fixed version
        
        Args:
            texts (list): List of text documents
            num_topics (int): Number of topics
            iterations (int): Number of iterations
            
        Returns:
            tp.CTModel: Trained CTM model
        """
        logger.info(f"Training CTM model with Tomotopy (k={num_topics})...")
        
        # Create CTM model
        ctm_model = tp.CTModel(k=num_topics)
        
        # Process and add documents properly
        for doc in texts:
            if isinstance(doc, str) and doc:
                # Split text into words
                words = [word for word in doc.lower().split() if len(word) >= 3]
                # Add document to the model
                if words:
                    ctm_model.add_doc(words)
        
        # Train the model
        logger.info(f"Training CTM model for {iterations} iterations...")
        for i in tqdm(range(0, iterations, 10)):
            ctm_model.train(10)
        
        # Store the model
        self.models['ctm'] = ctm_model
        
        # Get model results
        result = self.extract_tomotopy_results(ctm_model)
        self.model_results['ctm'] = result
        
        return ctm_model
    
    def train_pam_tomotopy(self, texts, num_topics=10, iterations=1000):
        """
        Train PAM (Pachinko Allocation Model) using Tomotopy - Fixed version
        
        Args:
            texts (list): List of text documents
            num_topics (int): Number of topics
            iterations (int): Number of iterations
            
        Returns:
            tp.PAModel: Trained PAM model
        """
        logger.info(f"Training PAM model with Tomotopy (k={num_topics})...")
        
        # Create PAM model - PAM requires both k1 and k2 parameters
        pam_model = tp.PAModel(k1=num_topics, k2=num_topics//2)  # k2 is typically half of k1
        
        # Process and add documents properly
        for doc in texts:
            if isinstance(doc, str) and doc:
                # Split text into words
                words = [word for word in doc.lower().split() if len(word) >= 3]
                # Add document to the model
                if words:
                    pam_model.add_doc(words)
        
        # Train the model
        logger.info(f"Training PAM model for {iterations} iterations...")
        for i in tqdm(range(0, iterations, 10)):
            pam_model.train(10)
        
        # Store the model
        self.models['pam'] = pam_model
        
        # Get model results
        result = self.extract_tomotopy_results(pam_model)
        self.model_results['pam'] = result
        
        return pam_model
    
    def train_ptm_tomotopy(self, texts, num_topics=10, iterations=1000):
        """
        Train PTM (Pitman-Yor Topic Model) using Tomotopy - Fixed version
        
        Args:
            texts (list): List of text documents
            num_topics (int): Number of topics
            iterations (int): Number of iterations
            
        Returns:
            tp.PTModel: Trained PTM model
        """
        logger.info(f"Training PTM model with Tomotopy (k={num_topics})...")
        
        # Create PTM model
        ptm_model = tp.PTModel(k=num_topics)
        
        # Process and add documents properly
        for doc in texts:
            if isinstance(doc, str) and doc:
                # Split text into words
                words = [word for word in doc.lower().split() if len(word) >= 3]
                # Add document to the model
                if words:
                    ptm_model.add_doc(words)
        
        # Train the model
        logger.info(f"Training PTM model for {iterations} iterations...")
        for i in tqdm(range(0, iterations, 10)):
            ptm_model.train(10)
        
        # Store the model
        self.models['ptm'] = ptm_model
        
        # Get model results
        result = self.extract_tomotopy_results(ptm_model)
        self.model_results['ptm'] = result
        
        return ptm_model
    
    def train_bertopic(self, texts, min_topic_size=5, nr_topics=10):
        """
        Train BERTopic model
        
        Args:
            texts (list): List of text documents
            min_topic_size (int): Minimum topic size
            nr_topics (int): Number of topics
            
        Returns:
            BERTopic: Trained BERTopic model
        """
        if not BERTOPIC_AVAILABLE:
            logger.warning("BERTopic not available. Skipping BERTopic training.")
            return None
            
        logger.info(f"Training BERTopic model (nr_topics={nr_topics})...")
        
        # Filter out empty texts
        filtered_texts = [text for text in texts if isinstance(text, str) and text.strip()]
        
        # Create vectorizer
        vectorizer = CountVectorizer(stop_words="english", min_df=5)
        
        # Create BERTopic model
        bertopic_model = BERTopic(
            language="english",
            calculate_probabilities=True,
            verbose=True,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            vectorizer_model=vectorizer
        )
        
        # Fit the model
        try:
            topics, probs = bertopic_model.fit_transform(filtered_texts)
            
            # Store the model
            self.models['bertopic'] = bertopic_model
            
            # Get model results
            result = {
                'topics': bertopic_model.get_topics(),
                'topic_info': bertopic_model.get_topic_info(),
                'document_info': list(zip(topics, probs)),
                'top_words': {topic: [word for word, _ in bertopic_model.get_topic(topic)] 
                             for topic in set(topics) if topic != -1}
            }
            self.model_results['bertopic'] = result
            
            return bertopic_model
        except Exception as e:
            logger.error(f"Error training BERTopic model: {e}")
            return None
    
    def extract_tomotopy_results(self, model, top_n=20):
        """
        Extract results from a Tomotopy model
        
        Args:
            model: Tomotopy model
            top_n (int): Number of top words to extract
            
        Returns:
            dict: Model results
        """
        # Get topics
        topics = {}
        for k in range(model.k):
            topic_words = model.get_topic_words(k, top_n=top_n)
            topics[k] = topic_words
        
        # Get document-topic distributions
        doc_topics = []
        for i in range(len(model.docs)):
            doc = model.docs[i]
            doc_topic_dist = {str(topic_id): weight for topic_id, weight in enumerate(doc.get_topic_dist())}
            doc_topics.append(doc_topic_dist)
        
        return {
            'topics': topics,
            'doc_topics': doc_topics,
            'perplexity': model.perplexity,
            'coherence': getattr(model, 'coherence', None),
            'top_words': {k: [word for word, _ in topics[k]] for k in topics}
        }
    
    def evaluate_models(self, texts, topics_range=range(5, 31, 5), iterations=1000):
        """
        Evaluate different topic models across a range of topic numbers
        
        Args:
            texts (list): List of text documents
            topics_range (range): Range of topic numbers to evaluate
            iterations (int): Number of iterations for training
            
        Returns:
            dict: Evaluation results
        """
        logger.info("Evaluating topic models across different topic numbers...")
        
        # Preprocess texts
        tokenized_docs, dictionary, corpus = self.preprocess_texts(texts)
        
        # Initialize results
        results = {
            'lda': {'perplexity': [], 'coherence': []},
            'ctm': {'perplexity': [], 'coherence': []},
            'pam': {'perplexity': [], 'coherence': []},
            'ptm': {'perplexity': [], 'coherence': []}
        }
        
        # Evaluate models for different numbers of topics
        for num_topics in topics_range:
            logger.info(f"Evaluating models with {num_topics} topics...")
            
            try:
                # Train LDA model
                lda_model = self.train_lda_tomotopy(texts, num_topics=num_topics, iterations=iterations)
                results['lda']['perplexity'].append(lda_model.perplexity)
            except Exception as e:
                logger.error(f"Error training LDA model: {e}")
                results['lda']['perplexity'].append(None)
            
            try:
                # Train CTM model
                ctm_model = self.train_ctm_tomotopy(texts, num_topics=num_topics, iterations=iterations)
                results['ctm']['perplexity'].append(ctm_model.perplexity)
            except Exception as e:
                logger.error(f"Error training CTM model: {e}")
                results['ctm']['perplexity'].append(None)
            
            try:
                # Train PAM model
                pam_model = self.train_pam_tomotopy(texts, num_topics=num_topics, iterations=iterations)
                results['pam']['perplexity'].append(pam_model.perplexity)
            except Exception as e:
                logger.error(f"Error training PAM model: {e}")
                results['pam']['perplexity'].append(None)
            
            try:
                # Train PTM model
                ptm_model = self.train_ptm_tomotopy(texts, num_topics=num_topics, iterations=iterations)
                results['ptm']['perplexity'].append(ptm_model.perplexity)
            except Exception as e:
                logger.error(f"Error training PTM model: {e}")
                results['ptm']['perplexity'].append(None)
            
            # Calculate coherence for each model
            for model_name in ['lda', 'ctm', 'pam', 'ptm']:
                if model_name in self.models and self.models[model_name] is not None:
                    try:
                        model = self.models[model_name]
                        topics = self.model_results[model_name]['top_words']
                        
                        # Generate list of topics' top words for coherence calculation
                        topics_list = [topics[topic_id] for topic_id in topics]
                        
                        # Calculate coherence using Gensim
                        coherence_model = CoherenceModel(
                            topics=topics_list,
                            texts=tokenized_docs,
                            dictionary=dictionary,
                            coherence='c_v'
                        )
                        coherence = coherence_model.get_coherence()
                        results[model_name]['coherence'].append(coherence)
                    except Exception as e:
                        logger.error(f"Error calculating coherence for {model_name}: {e}")
                        results[model_name]['coherence'].append(None)
                else:
                    results[model_name]['coherence'].append(None)
        
        # Store evaluation results
        self.evaluation = {
            'topics_range': list(topics_range),
            'results': results
        }
        
        return self.evaluation
    
    def plot_evaluation_results(self, filename="topic_model_evaluation.png"):
        """
        Plot evaluation results for different models
        
        Args:
            filename (str): Output filename
            
        Returns:
            str: Path to saved plot
        """
        if not self.evaluation:
            logger.warning("No evaluation results to plot")
            return None
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Get data
        topics_range = self.evaluation['topics_range']
        results = self.evaluation['results']
        
        model_colors = {
            'lda': 'blue',
            'ctm': 'green',
            'pam': 'red',
            'ptm': 'purple'
        }
        
        # Plot perplexity
        for model_name, color in model_colors.items():
            if model_name in results and 'perplexity' in results[model_name]:
                perplexity_values = results[model_name]['perplexity']
                if perplexity_values:
                    ax1.plot(topics_range, perplexity_values, 'o-', color=color, label=f"{model_name.upper()}")
        
        ax1.set_xlabel('Number of Topics')
        ax1.set_ylabel('Perplexity')
        ax1.set_title('Perplexity by Model and Number of Topics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot coherence
        for model_name, color in model_colors.items():
            if model_name in results and 'coherence' in results[model_name]:
                coherence_values = results[model_name]['coherence']
                if coherence_values and any(v is not None for v in coherence_values):
                    # Filter out None values
                    valid_indices = [i for i, v in enumerate(coherence_values) if v is not None]
                    valid_topics = [topics_range[i] for i in valid_indices]
                    valid_coherence = [coherence_values[i] for i in valid_indices]
                    ax2.plot(valid_topics, valid_coherence, 'o-', color=color, label=f"{model_name.upper()}")
        
        ax2.set_xlabel('Number of Topics')
        ax2.set_ylabel('Coherence Score (CV)')
        ax2.set_title('Topic Coherence by Model and Number of Topics')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation plot saved to {output_path}")
        return output_path
    
    def visualize_model_topics(self, model_name, output_base="topic_visualization"):
        """
        Visualize topics from a specific model
        
        Args:
            model_name (str): Name of the model to visualize
            output_base (str): Base name for output files
            
        Returns:
            str: Path to saved visualization
        """
        if model_name not in self.models or model_name not in self.model_results:
            logger.warning(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        results = self.model_results[model_name]
        
        # Handle different model types
        if model_name == 'bertopic' and BERTOPIC_AVAILABLE:
            # Use BERTopic's built-in visualization
            try:
                # Visualize topics
                fig = model.visualize_topics()
                output_path = os.path.join(self.output_dir, f"{output_base}_{model_name}_topics.html")
                fig.write_html(output_path)
                
                # Visualize topic hierarchy
                fig = model.visualize_hierarchy()
                hierarchy_path = os.path.join(self.output_dir, f"{output_base}_{model_name}_hierarchy.html")
                fig.write_html(hierarchy_path)
                
                # Visualize topic similarity
                fig = model.visualize_heatmap()
                heatmap_path = os.path.join(self.output_dir, f"{output_base}_{model_name}_heatmap.html")
                fig.write_html(heatmap_path)
                
                return output_path
            except Exception as e:
                logger.error(f"Error visualizing BERTopic model: {e}")
                return None
        else:
            # Visualize Tomotopy models
            return self.visualize_tomotopy_topics(model, results, model_name, output_base)
    
    def visualize_tomotopy_topics(self, model, results, model_name, output_base):
        """
        Visualize topics from a Tomotopy model
        
        Args:
            model: Tomotopy model
            results (dict): Model results
            model_name (str): Name of the model
            output_base (str): Base name for output files
            
        Returns:
            str: Path to saved visualization
        """
        # Get top words for each topic
        topics = results['topics']
        
        # Create a figure with subplots for each topic
        num_topics = len(topics)
        cols = 4
        rows = (num_topics + cols - 1) // cols  # Ceiling division
        
        plt.figure(figsize=(20, rows * 4))
        
        for i, topic_id in enumerate(topics):
            words, weights = zip(*topics[topic_id])
            
            # Create subplot
            ax = plt.subplot(rows, cols, i + 1)
            y_pos = np.arange(len(words[:10]))  # Top 10 words
            
            # Plot horizontal bar chart
            ax.barh(y_pos, weights[:10], align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words[:10])
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel('Weight')
            ax.set_title(f'Topic {topic_id}')
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, f"{output_base}_{model_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Topic visualization for {model_name} saved to {output_path}")
        return output_path
    
    def compare_topic_distributions(self, darkweb_texts, reddit_texts, model_name='lda', num_topics=10):
        """
        Compare topic distributions between Dark Web and Reddit
        
        Args:
            darkweb_texts (list): List of Dark Web texts
            reddit_texts (list): List of Reddit texts
            model_name (str): Name of the model to use
            num_topics (int): Number of topics
            
        Returns:
            tuple: (topic_comparison, plot_path)
        """
        logger.info(f"Comparing topic distributions using {model_name}...")
        
        # Train model on all texts
        all_texts = darkweb_texts + reddit_texts
        
        if model_name == 'lda':
            model = self.train_lda_tomotopy(all_texts, num_topics=num_topics)
        elif model_name == 'ctm':
            model = self.train_ctm_tomotopy(all_texts, num_topics=num_topics)
        elif model_name == 'pam':
            model = self.train_pam_tomotopy(all_texts, num_topics=num_topics)
        elif model_name == 'ptm':
            model = self.train_ptm_tomotopy(all_texts, num_topics=num_topics)
        elif model_name == 'bertopic' and BERTOPIC_AVAILABLE:
            model = self.train_bertopic(all_texts, nr_topics=num_topics)
        else:
            logger.error(f"Unknown model type: {model_name}")
            return None, None
        
        # Get topic distributions
        if model_name == 'bertopic' and BERTOPIC_AVAILABLE:
            # Handle BERTopic separately
            try:
                darkweb_topics, _ = model.transform(darkweb_texts)
                reddit_topics, _ = model.transform(reddit_texts)
                
                # Count topic distributions
                darkweb_distribution = Counter(darkweb_topics)
                reddit_distribution = Counter(reddit_topics)
                
                # Normalize distributions
                darkweb_total = sum(darkweb_distribution.values())
                reddit_total = sum(reddit_distribution.values())
                
                darkweb_normalized = {t: c/darkweb_total for t, c in darkweb_distribution.items()}
                reddit_normalized = {t: c/reddit_total for t, c in reddit_distribution.items()}
                
                # Prepare topic names using top words
                topic_info = model.get_topic_info()
                topic_names = {}
                
                for topic_id, row in topic_info.iterrows():
                    if topic_id != -1:  # Skip outlier topic
                        topic_words = [word for word, _ in model.get_topic(topic_id)][:3]
                        topic_names[topic_id] = f"Topic {topic_id}: {', '.join(topic_words)}"
                
                # Create comparison plot
                plot_path = self._plot_topic_comparison(
                    darkweb_normalized, reddit_normalized, topic_names, 
                    f"topic_comparison_{model_name}.png"
                )
                
                # Return topic distributions
                comparison = {
                    'darkweb_distribution': darkweb_normalized,
                    'reddit_distribution': reddit_normalized,
                    'topic_names': topic_names
                }
                
                return comparison, plot_path
                
            except Exception as e:
                logger.error(f"Error comparing BERTopic distributions: {e}")
                return None, None
        else:
            # Handle Tomotopy models
            # Get document-topic distributions from the model
            doc_topics = self.model_results[model_name]['doc_topics']
            
            # Split into darkweb and reddit
            darkweb_doc_topics = doc_topics[:len(darkweb_texts)]
            reddit_doc_topics = doc_topics[len(darkweb_texts):]
            
            # Aggregate topic distributions
            darkweb_distribution = {str(i): 0 for i in range(num_topics)}
            reddit_distribution = {str(i): 0 for i in range(num_topics)}
            
            for doc_topic in darkweb_doc_topics:
                for topic_id, weight in doc_topic.items():
                    darkweb_distribution[topic_id] += weight
            
            for doc_topic in reddit_doc_topics:
                for topic_id, weight in doc_topic.items():
                    reddit_distribution[topic_id] += weight
            
            # Normalize distributions
            darkweb_total = sum(darkweb_distribution.values())
            reddit_total = sum(reddit_distribution.values())
            
            darkweb_normalized = {t: w/darkweb_total for t, w in darkweb_distribution.items()}
            reddit_normalized = {t: w/reddit_total for t, w in reddit_distribution.items()}
            
            # Get topic names
            topic_names = {}
            for topic_id in range(num_topics):
                top_words = self.model_results[model_name]['top_words'][str(topic_id)][:3]
                topic_names[str(topic_id)] = f"Topic {topic_id}: {', '.join(top_words)}"
            
            # Create comparison plot
            plot_path = self._plot_topic_comparison(
                darkweb_normalized, reddit_normalized, topic_names, 
                f"topic_comparison_{model_name}.png"
            )
            
            # Return topic distributions
            comparison = {
                'darkweb_distribution': darkweb_normalized,
                'reddit_distribution': reddit_normalized,
                'topic_names': topic_names
            }
            
            return comparison, plot_path
    
    def _plot_topic_comparison(self, darkweb_dist, reddit_dist, topic_names, filename):
        """
        Plot comparison of topic distributions
        
        Args:
            darkweb_dist (dict): Dark Web topic distribution
            reddit_dist (dict): Reddit topic distribution
            topic_names (dict): Mapping of topic IDs to names
            filename (str): Output filename
            
        Returns:
            str: Path to saved plot
        """
        # Get common set of topics
        all_topics = sorted(set(darkweb_dist.keys()) | set(reddit_dist.keys()))
        
        if -1 in all_topics:  # Remove outlier topic from BERTopic
            all_topics.remove(-1)
        
        # Prepare data for plotting
        topics_list = []
        darkweb_values = []
        reddit_values = []
        
        for topic in all_topics:
            if topic in topic_names:
                topics_list.append(topic_names[topic])
            else:
                topics_list.append(f"Topic {topic}")
            
            darkweb_values.append(darkweb_dist.get(topic, 0))
            reddit_values.append(reddit_dist.get(topic, 0))
        
        # Create a figure
        plt.figure(figsize=(14, 10))
        
        # Define width of bars
        width = 0.35
        x = np.arange(len(topics_list))
        
        # Create bars
        plt.bar(x - width/2, darkweb_values, width, label='Dark Web', color='crimson')
        plt.bar(x + width/2, reddit_values, width, label='Reddit', color='royalblue')
        
        # Add labels, title, and legend
        plt.xlabel('Topics')
        plt.ylabel('Proportion')
        plt.title('Topic Distribution Comparison: Dark Web vs. Reddit')
        plt.xticks(x, topics_list, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Topic comparison saved to {output_path}")
        return output_path
    
    def get_best_model(self):
        """
        Determine the best model based on evaluation metrics
        
        Returns:
            tuple: (best_model_name, reasons)
        """
        if not self.evaluation or not self.evaluation.get('results'):
            logger.warning("No evaluation results to determine best model")
            return None, "No evaluation results available"
        
        results = self.evaluation['results']
        topics_range = self.evaluation['topics_range']
        
        # Gather metrics for each model
        metrics = {}
        reasons = []
        
        for model_name in results:
            if 'perplexity' in results[model_name] and 'coherence' in results[model_name]:
                # Find the best number of topics for this model
                perplexity_values = results[model_name]['perplexity']
                coherence_values = results[model_name]['coherence']
                
                # Find index with best (lowest) perplexity
                valid_perplexity = [v for v in perplexity_values if v is not None]
                if valid_perplexity:
                    valid_indices = [i for i, v in enumerate(perplexity_values) if v is not None]
                    best_perplexity_idx = valid_indices[np.argmin([perplexity_values[i] for i in valid_indices])]
                    best_perplexity = perplexity_values[best_perplexity_idx]
                    best_perplexity_topics = topics_range[best_perplexity_idx]
                else:
                    best_perplexity = None
                    best_perplexity_topics = None
                
                # Find index with best (highest) coherence
                valid_coherence = [v for v in coherence_values if v is not None]
                if valid_coherence:
                    valid_indices = [i for i, v in enumerate(coherence_values) if v is not None]
                    best_coherence_idx = valid_indices[np.argmax([coherence_values[i] for i in valid_indices])]
                    best_coherence = coherence_values[best_coherence_idx]
                    best_coherence_topics = topics_range[best_coherence_idx]
                else:
                    best_coherence = None
                    best_coherence_topics = None
                
                # Store metrics for this model
                metrics[model_name] = {
                    'best_perplexity': best_perplexity,
                    'best_perplexity_topics': best_perplexity_topics,
                    'best_coherence': best_coherence,
                    'best_coherence_topics': best_coherence_topics
                }
                
                if best_perplexity is not None and best_coherence is not None:
                    reasons.append(
                        f"{model_name.upper()}: Best perplexity {best_perplexity:.2f} at {best_perplexity_topics} topics, " +
                        f"best coherence {best_coherence:.4f} at {best_coherence_topics} topics")
        
        # Determine the best model
        # We prioritize coherence as it's more closely related to human interpretability
        best_model = None
        best_coherence = -float('inf')
        
        for model_name, model_metrics in metrics.items():
            if model_metrics['best_coherence'] and model_metrics['best_coherence'] > best_coherence:
                best_coherence = model_metrics['best_coherence']
                best_model = model_name
        
        if best_model:
            best_topics = metrics[best_model]['best_coherence_topics']
            reasons.insert(0,
                           f"Best model overall: {best_model.upper()} with {best_topics} topics (coherence: {best_coherence:.4f})")
            return best_model, reasons
        else:
            return None, "Could not determine best model based on available metrics"
            
    def generate_topic_model_report(self, output_file="topic_modeling_report.md"):
        """
        Generate a comprehensive report on topic modeling results

        Args:
            output_file (str): Output filename

        Returns:
            str: Path to report file
        """
        # Create report content
        report = """
        # Topic Modeling Analysis Report

        ## Overview

        This report presents the results of applying multiple topic modeling algorithms to OpSec discourse data.

        ## Models Evaluated

        - **LDA (Latent Dirichlet Allocation)**: A classic probabilistic topic model.
        - **CTM (Correlated Topic Model)**: Extends LDA by allowing topics to be correlated.
        - **PAM (Pachinko Allocation Model)**: Captures correlations between topics using a directed acyclic graph.
        - **PTM (Pitman-Yor Topic Model)**: Uses Pitman-Yor process instead of Dirichlet for word distributions.
        - **BERTopic**: Leverages BERT embeddings with clustering techniques for topic identification.

        ## Evaluation Metrics

        - **Perplexity**: Measures how well a model predicts unseen data. Lower values are better.
        - **Coherence (CV)**: Measures semantic similarity among top words in topics. Higher values are better.
        """

        # Add evaluation results if available
        if self.evaluation:
            report += """
            ## Evaluation Results

            The following models were evaluated across a range of topic numbers:
            """

            # Get best model and reasons
            best_model, reasons = self.get_best_model()

            if best_model:
                report += "\n**Best Model:**\n\n"
                for reason in reasons:
                    report += f"- {reason}\n"

                report += "\n### Perplexity and Coherence by Number of Topics\n\n"
                report += "![Topic Model Evaluation](topic_model_evaluation.png)\n"

            # Add model details
            for model_name in self.models:
                if model_name in self.model_results:
                    report += f"\n## {model_name.upper()} Model Details\n\n"

                    # Add basic model info
                    if model_name == 'bertopic':
                        # BERTopic-specific info
                        topic_info = self.model_results[model_name].get('topic_info')
                        if topic_info is not None:
                            report += "### Topic Information\n\n"
                            report += "| Topic ID | Count | Name |\n"
                            report += "|---------|-------|------|\n"

                            for _, row in topic_info.iterrows():
                                if row['Topic'] != -1:  # Skip outlier topic
                                    report += f"| {row['Topic']} | {row['Count']} | {row['Name']} |\n"
                    else:
                        # Tomotopy model info
                        topics = self.model_results[model_name].get('top_words', {})

                        report += "### Topics and Top Words\n\n"
                        report += "| Topic ID | Top Words |\n"
                        report += "|---------|----------|\n"

                        for topic_id in sorted(topics.keys()):
                            top_words = topics[topic_id][:10]
                            report += f"| {topic_id} | {', '.join(top_words)} |\n"

                    # Add visualization reference
                    report += f"\n### Topic Visualization\n\n"
                    report += f"![{model_name.upper()} Topics](topic_visualization_{model_name}.png)\n"

            # Add topic distribution comparison
            if hasattr(self, 'topic_comparison') and self.topic_comparison:
                report += """
                ## Topic Distribution Comparison

                The following visualization shows the distribution of topics between Dark Web and Reddit discussions:

                ![Topic Distribution Comparison](topic_comparison_lda.png)

                This comparison helps identify differences in discourse focus between the two platforms.
                """

        # Add conclusion
        report += """
        ## Conclusion

        The topic modeling analysis reveals distinct patterns in OpSec discourse, highlighting key themes
        and differences between Dark Web and Reddit discussions. The best model based on coherence scores
        provides the most interpretable representation of topics in the corpus.
        """

        # Clean up report formatting
        report = "\n".join([line.lstrip() for line in report.split("\n")])

        # Save report
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Topic modeling report saved to {output_path}")
        return output_path

def run_topic_modeling(darkweb_docs, reddit_docs, output_dir):
    """
    Run topic modeling analysis on the provided documents
    
    Args:
        darkweb_docs (list): List of Dark Web documents
        reddit_docs (list): List of Reddit documents
        output_dir (str): Output directory for results
        
    Returns:
        tuple: (modeler, report_path)
    """
    # Initialize modeler
    modeler = AdvancedTopicModeler(output_dir=output_dir)
    
    # Combine documents for evaluation
    all_docs = darkweb_docs + reddit_docs
    
    # Evaluate models
    evaluation = modeler.evaluate_models(all_docs)
    
    # Get best model and optimal number of topics
    best_model, reasons = modeler.get_best_model()
    if best_model:
        # Get optimal number of topics from evaluation
        optimal_topics = modeler.evaluation['results'][best_model]['best_coherence_topics']
        
        # Train the best model
        if best_model == 'lda':
            modeler.train_lda_tomotopy(all_docs, num_topics=optimal_topics)
        elif best_model == 'ctm':
            modeler.train_ctm_tomotopy(all_docs, num_topics=optimal_topics)
        elif best_model == 'pam':
            modeler.train_pam_tomotopy(all_docs, num_topics=optimal_topics)
        elif best_model == 'ptm':
            modeler.train_ptm_tomotopy(all_docs, num_topics=optimal_topics)
        
        # Visualize topics
        modeler.visualize_model_topics(best_model)
        
        # Compare topic distributions between Dark Web and Reddit
        comparison, plot_path = modeler.compare_topic_distributions(
            darkweb_docs, reddit_docs, model_name=best_model, num_topics=optimal_topics
        )
        
        if comparison:
            modeler.topic_comparison = comparison
        
        # Try BERTopic for comparison regardless of best model
        try:
            logger.info("Training BERTopic model for comparison...")
            modeler.train_bertopic(all_docs, nr_topics=optimal_topics)
            modeler.visualize_model_topics('bertopic')
        except Exception as e:
            logger.error(f"Error training BERTopic model: {e}")
    
    # Generate report
    report_path = modeler.generate_topic_model_report()
    
    return modeler, report_path

if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Topic Modeling for OpSec Analysis")
    parser.add_argument("--darkweb-dir", default="data/darkweb", help="Directory containing Dark Web data")
    parser.add_argument("--reddit-dir", default="data/reddit", help="Directory containing Reddit data")
    parser.add_argument("--output-dir", default="analysis_results/topic_models", help="Output directory")

    args = parser.parse_args()

    # Load data
    import glob
    import json

    darkweb_files = glob.glob(os.path.join(args.darkweb_dir, "*.json"))
    reddit_files = glob.glob(os.path.join(args.reddit_dir, "*.json"))

    darkweb_data = []
    reddit_data = []

    for file_path in darkweb_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                darkweb_data.extend(data)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")

    for file_path in reddit_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                reddit_data.extend(data)
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")

    # Extract content
    darkweb_docs = []
    reddit_docs = []

    for item in darkweb_data:
        if isinstance(item, dict) and 'content' in item:
            darkweb_docs.append(item['content'])

    for item in reddit_data:
        content_field = 'content'
        if isinstance(item, dict):
            if 'content' in item:
                reddit_docs.append(item['content'])
            elif 'selftext' in item:
                reddit_docs.append(item['selftext'])

    # Run topic modeling
    modeler, report_path = run_topic_modeling(darkweb_docs, reddit_docs, args.output_dir)

    if report_path:
        logger.info(f"Topic modeling completed successfully. Report saved to {report_path}")
    else:
        logger.error("Topic modeling failed to complete successfully.")