#!/usr/bin/env python
"""
Fixed OpSec Discourse Analysis Pipeline
---------------------------------
This script runs the complete OpSec discourse analysis pipeline:
1. Scrapes data from SuprBay (Dark Web) and Reddit
2. Analyzes the collected data
3. Generates visualizations and comparative analysis report

Fixed the following issues:
1. String indices error in Reddit scraper
2. Unrecognized argument in analyzer
"""

import os
import sys
import argparse
import subprocess
import logging
import time
from datetime import datetime
import random
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("opsec_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("opsec_pipeline")


def check_requirements():
    """Check if all required tools and dependencies are installed"""
    logger.info("Checking requirements...")
    
    # Check for required Python libraries
    required_libs = [
        'requests', 'bs4', 'nltk', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'sklearn', 'wordcloud', 'socks'
    ]
    
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    
    if missing_libs:
        logger.error(f"Missing required libraries: {', '.join(missing_libs)}")
        logger.error("Install them with: pip install " + " ".join(missing_libs))
        return False
    
    # Check if Tor is running (if we're scraping Dark Web)
    if not args.skip_darkweb:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', args.tor_port))
            if result != 0:
                logger.error(f"Tor SOCKS proxy not detected on port {args.tor_port}")
                logger.error("Make sure Tor Browser is running before scraping Dark Web forums")
                return False
            sock.close()
        except Exception as e:
            logger.error(f"Error checking Tor connection: {e}")
            return False
    
    logger.info("All requirements satisfied")
    return True


def create_directories():
    """Create necessary directories for the pipeline"""
    dirs = [
        args.data_dir,
        os.path.join(args.data_dir, "darkweb"),
        os.path.join(args.data_dir, "reddit"),
        args.output_dir,
        os.path.join(args.output_dir, "visualizations")
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Created directory: {d}")


def scrape_suprbay():
    """Run the SuprBay scraper"""
    if args.skip_darkweb:
        logger.info("Skipping Dark Web scraping")
        return True
    
    logger.info("Starting SuprBay scraping...")
    
    # Get SuprBay configuration from config
    config = load_config(args.config)
    suprbay_config = next((target for target in config['targets'] if target['name'] == 'SuprBay'), None)
    
    if not suprbay_config or not suprbay_config['urls']:
        logger.error("No SuprBay configuration found in config file")
        return False
    
    forum_url = suprbay_config['urls'][0]  # Use the first URL from the config
    
    cmd = [
        sys.executable, "suprbay_scraper.py",
        "--forum-section", forum_url,
        "--output-dir", os.path.join(args.data_dir, "darkweb"),
        "--tor-port", str(args.tor_port)
    ]
    
    if args.debug:
        cmd.append("--debug")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("SuprBay scraping completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during SuprBay scraping: {e}")
        return False


def scrape_reddit(config, skip_reddit=False):
    """
    Scrape data from Reddit subreddits using the fixed scraper

    Args:
        config (dict or str): Configuration dictionary or path to config file
        skip_reddit (bool): Whether to skip Reddit scraping

    Returns:
        bool: True if successful, False otherwise
    """
    if skip_reddit:
        logger.info("Skipping Reddit scraping as requested")
        return True

    try:
        # Import the RedditScraper from opsec_scraper
        from opsec_scraper import RedditScraper
        
        # Get Reddit configuration
        if isinstance(config, str):
            # If config is a file path, load it
            with open(config, 'r') as f:
                config = json.load(f)
        
        reddit_config = None
        for target in config.get('targets', []):
            if target.get('name') == 'Reddit':
                reddit_config = target
                break
        
        if not reddit_config or not reddit_config.get('subreddits'):
            logger.warning("No Reddit configuration found in config file")
            return False

        # Create output directory if it doesn't exist
        output_dir = os.path.join('data', 'reddit')
        os.makedirs(output_dir, exist_ok=True)

        # Initialize the Reddit scraper
        scraper = RedditScraper(output_dir=output_dir)

        # Get scraping parameters
        subreddits = reddit_config.get('subreddits', [])
        scraping_params = reddit_config.get('scraping_params', {})
        limit = scraping_params.get('limit', 100)
        time_filter = scraping_params.get('time_filter', 'month')

        # Scrape each subreddit
        for subreddit in subreddits:
            try:
                logger.info(f"Scraping subreddit: r/{subreddit}")
                scraper.scrape_subreddit(
                    subreddit=subreddit,
                    limit=limit,
                    time_filter=time_filter
                )
                # Add a delay between subreddits
                time.sleep(random.uniform(5, 10))
            except Exception as e:
                logger.error(f"Error scraping subreddit r/{subreddit}: {e}")
                continue

        logger.info("Reddit scraping completed")
        return True

    except Exception as e:
        logger.error(f"Error in Reddit scraping pipeline: {e}")
        return False


def run_analysis():
    """Run the data analysis with fixed arguments"""
    if args.skip_analysis:
        logger.info("Skipping data analysis")
        return True
    
    logger.info("Starting data analysis...")
    
    # Fix: remove the --darkweb-name argument that's causing the error
    cmd = [
        sys.executable, "opsec_analyzer.py",
        "--darkweb-dir", os.path.join(args.data_dir, "darkweb"),
        "--reddit-dir", os.path.join(args.data_dir, "reddit"),
        "--output-dir", args.output_dir,
        "--report", args.report_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Data analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during data analysis: {e}")
        return False


def run_pipeline():
    """Run the full pipeline"""
    logger.info("Starting OpSec Discourse Analysis Pipeline")
    
    # Check requirements
    if not check_requirements():
        logger.error("Requirements check failed. Please fix the issues and try again.")
        return False
    
    # Create directories
    create_directories()
    
    # Step 1: Scrape data
    suprbay_success = scrape_suprbay()
    reddit_success = scrape_reddit(args.config, args.skip_reddit)
    
    if not (suprbay_success or reddit_success):
        logger.error("All scraping steps failed. Cannot proceed to analysis.")
        return False
    
    # Step 2: Analyze data
    analysis_success = run_analysis()
    
    if not analysis_success:
        logger.error("Analysis failed.")
        return False
    
    logger.info("Pipeline completed successfully!")
    logger.info(f"Analysis results and report saved to {args.output_dir}")
    
    return True


def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        raise


def main():
    """Main function to run the OpSec discourse analysis pipeline."""
    parser = argparse.ArgumentParser(description='Run OpSec discourse analysis pipeline')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data-dir', default='data', help='Directory to store raw data')
    parser.add_argument('--output-dir', default='analysis_results', help='Directory to store analysis results')
    parser.add_argument('--tor-port', type=int, default=9150, help='Tor SOCKS proxy port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--skip-darkweb', action='store_true', help='Skip Dark Web scraping')
    parser.add_argument('--skip-reddit', action='store_true', help='Skip Reddit scraping')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip data analysis')
    parser.add_argument('--report-file', default='opsec_comparative_analysis.md', help='Output report filename')
    
    global args
    args = parser.parse_args()

    # Run the pipeline
    success = run_pipeline()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()