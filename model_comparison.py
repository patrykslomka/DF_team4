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
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forum_analyzer")

class ModelComparison:
  def __init__(self, input_file="processed_data.json", output_dir="model_comparison"):
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
    
  
  def model_topics(self, texts: List[str], ):
    