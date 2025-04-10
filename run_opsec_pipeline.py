#!/usr/bin/env python
"""
OpSec Discourse Analysis Pipeline
---------------------------------
This script runs the complete OpSec discourse analysis pipeline:
1. Scrapes data from Dark Web (SuprBay) and Surface Web (Reddit) forums
2. Analyzes the collected data
3. Generates visualizations and comparative analysis report
"""

import os
import sys
import argparse
import subprocess
import logging
import time
from datetime import datetime

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
        'seaborn', 'scikit-learn', 'wordcloud', 'PySocks'
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
        os.path.join(args.data_dir, "surfaceweb"),
        os.path.join(args.data_dir, "reddit"),
        args.output_dir,
        os.path.join(args.output_dir, "visualizations")
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Created directory: {d}")


def scrape_darkweb():
    """Run the Dark Web scraper"""
    if args.skip_darkweb:
        logger.info("Skipping Dark Web scraping")
        return True

    logger.info("Starting Dark Web scraping...")

    # Check if thread list file exists
    if not os.path.exists(args.thread_list):
        logger.error(f"Thread list file not found: {args.thread_list}")
        return False

    cmd = [
        sys.executable, "suprbay_scraper.py",
        "--thread-list", args.thread_list,
        "--output-dir", os.path.join(args.data_dir, "darkweb"),
        "--tor-port", str(args.tor_port),
        "--max-pages", str(args.max_pages)
    ]

    if args.debug:
        cmd.append("--debug")

    try:
        subprocess.run(cmd, check=True)
        logger.info("Dark Web scraping completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during Dark Web scraping: {e}")
        return False


def scrape_reddit():
    """Run the Reddit scraper"""
    if args.skip_reddit:
        logger.info("Skipping Reddit scraping")
        return True

    logger.info("Starting Reddit scraping...")

    # Split subreddits string into a list
    subreddits = [s.strip() for s in args.subreddits.split(',')]

    success = True
    for subreddit in subreddits:
        logger.info(f"Scraping subreddit: r/{subreddit}")

        cmd = [
            sys.executable, "opsec_scraper.py",
            "--reddit",
            "--subreddit", subreddit,
            "--reddit-limit", str(args.reddit_limit),
            "--output-dir", args.data_dir
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully scraped r/{subreddit}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error scraping r/{subreddit}: {e}")
            success = False

        # Add delay between subreddits to avoid rate limiting
        if subreddit != subreddits[-1]:
            delay = 5
            logger.info(f"Waiting {delay} seconds before scraping next subreddit...")
            time.sleep(delay)

    return success


def run_analysis():
    """Run the data analysis"""
    if args.skip_analysis:
        logger.info("Skipping data analysis")
        return True

    logger.info("Starting data analysis...")

    cmd = [
        sys.executable, "suprbay_analyzer.py",
        "--darkweb-dir", os.path.join(args.data_dir, "darkweb"),
        "--surfaceweb-dir", os.path.join(args.data_dir, "surfaceweb"),
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
    darkweb_success = scrape_darkweb()
    reddit_success = scrape_reddit()

    if not darkweb_success and not reddit_success:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpSec Discourse Analysis Pipeline")

    # General arguments
    parser.add_argument("--data-dir", default="data", help="Base directory for data storage")
    parser.add_argument("--output-dir", default="analysis_results", help="Directory for analysis output")
    parser.add_argument("--report-file", default="opsec_comparative_analysis.md", help="Output report filename")

    # Dark Web arguments
    parser.add_argument("--skip-darkweb", action="store_true", help="Skip Dark Web scraping")
    parser.add_argument("--tor-port", type=int, default=9150, help="Tor SOCKS proxy port")
    parser.add_argument("--thread-list", default="suprbay_threads.txt", help="File containing SuprBay thread URLs")
    parser.add_argument("--max-pages", type=int, default=5, help="Maximum number of pages to scrape per thread")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for scrapers")

    # Surface Web arguments
    parser.add_argument("--skip-reddit", action="store_true", help="Skip Reddit scraping")
    parser.add_argument("--subreddits", default="opsec,privacy,cybersecurity",
                        help="Comma-separated list of subreddits to scrape")
    parser.add_argument("--reddit-limit", type=int, default=100,
                        help="Maximum number of Reddit posts to scrape per subreddit")

    # Analysis arguments
    parser.add_argument("--skip-analysis", action="store_true", help="Skip data analysis")

    args = parser.parse_args()

    # Run the pipeline
    success = run_pipeline()

    if not success:
        sys.exit(1)