# Forum Discourse Analysis: Dark Web & Reddit

## Overview
This project analyzes and compares discussions from Dark Web forums and Reddit using advanced topic modeling and NLP techniques. It includes scraping, preprocessing, topic modeling (LDA, NMF, KMeans, BERTopic), and rich visualizations to reveal key themes and differences between the two sources.

## Project Structure
```
├── config.json                    # Configuration for scraping and analysis parameters
├── opsec_scraper.py               # Reddit scraper for security/privacy subreddits
├── suprbay_scraper.py             # Dark web scraper for SuprBay forum
├── preprocess.py                  # Text preprocessing and cleaning
├── forum_analyzer.py              # Main analysis script with all topic models
├── demo.py                        # Streamlit web interface for exploration
├── model_comparison.ipynb         # Jupyter notebook for detailed model comparison
├── processed_data.json            # Preprocessed data output (generated)
├── opsec_scrape.csv              # Sample output file (generated)
├── data/                          # Raw scraped data directory
│   ├── reddit/                    # Reddit posts
│   └── darkweb/                   # Dark web forum posts
├── analysis_results/              # Analysis outputs
│   └── visualizations/            # Generated charts and plots
├── model_comparison/              # Model comparison outputs
│   └── best_models/               # Saved topic models
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker container configuration
├── docker-compose.yml             # Docker compose for easy deployment
└── README.md                      # This file
```

## Data Sources
- **Dark Web:** Scraped from SuprBay (The Pirate Bay's forum) and similar sources using Tor
- **Reddit:** Scraped from security/privacy-related subreddits using Reddit API

## Pipeline
1. **Scraping:**
   - `opsec_scraper.py` and `suprbay_scraper.py` collect posts and threads from Reddit and Dark Web forums
2. **Preprocessing:**
   - `preprocess.py` cleans, tokenizes, removes stopwords, and lemmatizes text
   - Output: `processed_data.json`
3. **Topic Modeling & Analysis:**
   - `forum_analyzer.py` runs four topic modeling techniques (LDA, NMF, KMeans, BERTopic) on both sources
   - Extracts and visualizes the top 10 most important words for each model/source
   - Generates topic prevalence plots and word clouds for context
4. **Model Comparison:**
   - `model_comparison.ipynb` provides detailed Jupyter notebook analysis with coherence metrics
5. **Visualizations:**
   - All results are saved in `analysis_results/visualizations/` as interactive HTML and PNG files
6. **Interactive Demo:**
   - `demo.py` provides a Streamlit web interface to explore and compare topics from both sources

## Requirements
- **Python:** 3.8–3.11 (tested with Python 3.10)
- **Tor Browser:** Required for Dark Web scraping (must be running on port 9150)
- **Memory:** At least 4GB RAM recommended for BERTopic analysis
- **Storage:** ~2GB free space for models and data

## Setup & Installation

### Option 1: Native (Local) Setup
1. **Clone the repository and navigate to the project folder:**
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLTK data:**
   ```python
   python -c "
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   "
   ```

5. **Verify installation:**
   ```bash
   python -c "import bertopic, sentence_transformers, streamlit; print('All dependencies installed successfully!')"
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

### Quick Start (Using Existing Data)
If you want to skip scraping and use the existing `processed_data.json`:

```bash
# Run the main analysis
python forum_analyzer.py

# Start the interactive demo
streamlit run demo.py

# Open the Jupyter notebook for detailed analysis
jupyter notebook model_comparison.ipynb
```

### Full Pipeline (Including Scraping)

### 1. Configure Scraping (Optional)
Edit `config.json` to customize:
- Subreddits to scrape
- Keywords to search for
- Number of topics for analysis
- Custom stopwords

### 2. Scrape Data
**For Reddit:**
```bash
python opsec_scraper.py
```

**For Dark Web (requires Tor):**
1. Start Tor Browser (must be running on port 9150)
2. Run the scraper:
```bash
python suprbay_scraper.py
```

### 3. Preprocess Data
```bash
python preprocess.py
```
This combines and cleans all scraped data into `processed_data.json`

### 4. Run Topic Analysis
```bash
python forum_analyzer.py
```
Results will be saved in `analysis_results/visualizations/`:
- `bar_topwords_darkweb_{MODEL}.html` - Top words for dark web posts
- `bar_topwords_reddit_{MODEL}.html` - Top words for Reddit posts  
- `wordcloud_{source}_{MODEL}.png` - Word clouds for each source/model
- `topic_prevalence_{MODEL}.html` - Topic distribution plots

### 5. Interactive Exploration
```bash
streamlit run demo.py
```
Open http://localhost:8501 to explore topics interactively

### 6. Detailed Model Comparison
```bash
jupyter notebook model_comparison.ipynb
```
This notebook provides:
- Coherence score comparisons
- Model performance metrics
- Detailed visualizations
- Saved best-performing models

## Configuration
The `config.json` file controls:
- **Tor settings:** Proxy configuration for dark web scraping
- **Target sources:** URLs and subreddits to scrape
- **Analysis parameters:** Number of topics, keywords, stopwords
- **Scraping limits:** Pages, threads, and delay settings

## Output Files
- **Data files:**
  - `processed_data.json` - Cleaned and processed text data
  - `data/reddit/` and `data/darkweb/` - Raw scraped data
  
- **Analysis results:**
  - `analysis_results/visualizations/` - Charts, plots, and word clouds
  - `model_comparison/best_models/` - Saved topic models
  
- **Interactive outputs:**
  - Streamlit app (accessible via web browser)
  - Jupyter notebook with detailed analysis

## Troubleshooting

### Common Issues:
1. **NLTK download errors:** Run the NLTK download commands manually
2. **Tor connection issues:** Ensure Tor Browser is running on port 9150
3. **Memory errors with BERTopic:** Reduce the dataset size or use a machine with more RAM
4. **Missing packages:** Run `pip install -r requirements.txt` again

### For Windows Users:
- Use PowerShell or Command Prompt
- Ensure Python is added to PATH
- May need to install Microsoft Visual C++ Build Tools for some packages

### For macOS/Linux Users:
- May need to install additional system dependencies for some packages
- Use `python3` instead of `python` if needed

## Notes
- **Dark Web scraping requires Tor:** Download and run Tor Browser before scraping
- **Rate limiting:** Scrapers include delays to be respectful to servers
- **Data privacy:** All scraped data is for research purposes only
- **Modular design:** You can run individual components as needed
- **Cross-platform:** Works on Windows, macOS, and Linux

## License
This project is for educational and research purposes only.