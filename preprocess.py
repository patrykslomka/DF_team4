import json
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Union
import os
from pathlib import Path

# NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Custom stop words specific to forums
        custom_stop_words = {
            'post', 'thread', 'user', 'comment', 'op', 'reddit', 'subreddit',
            'edit', 'update', 'deleted', 'removed', 'http', 'https', 'www',
            'com', 'org', 'net', 'gov', 'edu', 'html', 'php', 'jpg', 'png',
            'gif', 'img', 'image', 'link', 'url', 'quote', 'spoiler', 'nsfw',
            'tldr', 'tl;dr', 'afaik', 'imo', 'imho', 'fyi', 'ftw', 'irl',
            'ama', 'eli5', 'til', 'dae', 'mfw', 'mrw', 'smh', 'tbh', 'yolo'
        }
        self.stop_words.update(custom_stop_words)

    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text."""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()

    def remove_urls_and_emails(self, text: str) -> str:
        """Remove URLs and email addresses from text."""
        # Remove URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(url_pattern, '', text)
        
        # Remove email addresses
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        text = re.sub(email_pattern, '', text)
        
        return text

    def handle_special_characters(self, text: str) -> str:
        """Handle special characters and punctuation."""
        # Replace special characters with space
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Standardize apostrophes and hyphens
        text = text.replace('’', "'").replace('"', '"').replace('"', '"')
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def handle_emojis(self, text: str) -> str:
        """Convert emojis to text or remove them."""
        # Convert emojis to text representation
        text = emoji.demojize(text)
        
        # Remove emoji text representations
        text = re.sub(r':[a-z_]+:', '', text)
        
        return text

    def preprocess_text(self, text: str) -> str:
        """Apply all preprocessing steps to a single text."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Apply preprocessing steps
        text = self.remove_html_tags(text)
        text = self.remove_urls_and_emails(text)
        text = self.handle_special_characters(text)
        text = self.handle_emojis(text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize - Reduce words to their base or dictionary form (e.g. running -> run)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

    def process_json_file(self, file_path: str, source: str) -> List[Dict]:
        """Process a JSON file containing forum posts."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_posts = []

        # Accept both list and dict (with 'data' key)
        if isinstance(data, dict) and 'data' in data:
            posts = data['data']
        elif isinstance(data, list):
            posts = data
        else:
            posts = []

        for post in posts:
            if not isinstance(post, dict):
                continue
            # Use 'content', fallback to 'selftext', 'body', 'title'
            raw_text = post.get('content', '') or post.get('selftext', '') or post.get('body', '') or post.get('title', '')
            processed_text = self.preprocess_text(raw_text)
            # Skip posts with very short preprocessed text (less than 3 words)
            if processed_text and len(processed_text.split()) >= 3:
                processed_posts.append({
                    'id': post.get('id', ''),
                    'text': processed_text,
                    'source': source
                })
        return processed_posts

def main():
    preprocessor = TextPreprocessor()
    processed_data = []
    # Process Reddit data
    reddit_dir = Path('data/reddit')
    for file in reddit_dir.glob('*.json'):
        try:
            processed_posts = preprocessor.process_json_file(str(file), source='reddit')
            processed_data.extend(processed_posts)
            print(f"Processed {file.name}: {len(processed_posts)} posts")
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
    # Process Dark Web data
    darkweb_dir = Path('data/darkweb')
    for file in darkweb_dir.glob('*.json'):
        try:
            processed_posts = preprocessor.process_json_file(str(file), source='darkweb')
            processed_data.extend(processed_posts)
            print(f"Processed {file.name}: {len(processed_posts)} posts")
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
    # Save processed data
    output_file = 'processed_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"\nTotal processed posts: {len(processed_data)}")
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main() 