# Forum Discourse Analysis: Dark Web & Reddit

## Overview
This project analyzes and compares discussions from Dark Web forums and Reddit using advanced topic modeling and NLP techniques. It includes scraping, preprocessing, topic modeling (LDA, NMF, KMeans, BERTopic), and rich visualizations to reveal key themes and differences between the two sources.

## Data Sources
- **Dark Web:** Scraped from SuprBay (The Pirate Bay's forum) and similar sources.
- **Reddit:** Scraped from security/privacy-related subreddits.

## Pipeline
1. **Scraping:**
   - `opsec_scraper.py` and `suprbay_scraper.py` collect posts and threads from Reddit and Dark Web forums.
2. **Preprocessing:**
   - `preprocess.py` cleans, tokenizes, removes stopwords, and lemmatizes text. Output: `processed_data.json`.
3. **Topic Modeling & Analysis:**
   - `forum_analyzer.py` runs four topic modeling techniques (LDA, NMF, KMeans, BERTopic) on both sources.
   - Extracts and visualizes the top 10 most important words for each model/source.
   - Generates topic prevalence plots and word clouds for context.
4. **Visualizations:**
   - All results are saved in `analysis_results/visualizations/` as interactive HTML and PNG files.
5. **Interactive Demo:**
   - `demo.py` provides a Streamlit web interface to explore and compare topics from both sources.

## Setup
### Option 1: Native (Local) Setup
1. **Clone the repository and navigate to the project folder.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **(Optional) Download NLTK data:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```
4. **Run the Streamlit demo:**
   ```bash
   streamlit run demo.py
   ```

### Option 2: Docker (Recommended for Reproducibility)
1. **Build and start the container with the Streamlit app:**
   ```bash
   docker-compose up --build
   ```
   Access the Streamlit interface at http://localhost:8501

2. **For interactive development in the container:**
   ```bash
   # In a new terminal while the container is running
   docker-compose exec forum-analysis bash
   ```

3. **Run specific scripts in the container:**
   ```bash
   # Inside the container
   python preprocess.py
   python forum_analyzer.py
   # etc.
   ```

## Usage
### 1. Scrape Data
- **Reddit:**
  ```bash
  python opsec_scraper.py
  ```
- **Dark Web:**
  ```bash
  python suprbay_scraper.py
  ```

### 2. Preprocess Data
```bash
python preprocess.py
```

### 3. Analyze & Visualize
```bash
python forum_analyzer.py
```
- Results: See `analysis_results/visualizations/` for bar charts, word clouds, and topic prevalence plots.

### 4. Interactive Exploration
```bash
streamlit run demo.py
```
- The Streamlit app lets you search for topics and compare how they appear in Dark Web forums vs. Reddit.

## Main Outputs
- `bar_topwords_darkweb_{MODEL}.html` and `bar_topwords_reddit_{MODEL}.html`: Top 10 words for each model/source.
- `wordcloud_{source}_{MODEL}.png`: Word clouds for each model/source.
- `topic_prevalence_{MODEL}.html`: Topic prevalence plots.
- Interactive Streamlit demo for exploring and comparing topics.

## Notes
- Requires Python 3.8–3.11 (Docker uses Python 3.11).
- For Dark Web scraping, Tor must be running (see suprbay_scraper.py for details).
- All scripts are modular; you can run only the parts you need.