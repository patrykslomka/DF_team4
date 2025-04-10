"""
SuprBay Thread Scraper
----------------------
A specialized scraper for the SuprBay Dark Web forum.
This script will scrape specific OpSec discussion threads.
"""

import os
import time
import json
import random
import argparse
import logging
import re
from datetime import datetime
from urllib.parse import urlparse, urljoin

# Web scraping libraries
import requests
from bs4 import BeautifulSoup

# For Tor connectivity
import socks
import socket

# NLP libraries for basic text processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("suprbay_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("suprbay_scraper")

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class SuprBayScraper:
    """Specialized scraper for SuprBay forum threads"""

    def __init__(self, output_dir="data/darkweb", proxy_port=9150, delay=(8, 15), debug=False):
        """
        Initialize the SuprBay scraper

        Args:
            output_dir (str): Directory to store scraped data
            proxy_port (int): Tor SOCKS proxy port (usually 9150 for Tor Browser)
            delay (tuple): Random delay range between requests (min, max) in seconds
            debug (bool): Whether to save HTML for debugging purposes
        """
        self.output_dir = output_dir
        self.delay = delay
        self.proxy_port = proxy_port
        self.debug = debug

        # Create debug directory if needed
        if self.debug:
            self.debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(self.debug_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set up Tor connection
        self.setup_tor_connection()

    def setup_tor_connection(self):
        """Configure the session to route through Tor"""
        # Set up the SOCKS proxy for Tor
        socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", self.proxy_port)
        socket.socket = socks.socksocket

        # Configure requests to use the SOCKS proxy
        self.session.proxies = {
            'http': f'socks5h://127.0.0.1:{self.proxy_port}',
            'https': f'socks5h://127.0.0.1:{self.proxy_port}'
        }

        logger.info(f"Tor connection established using port {self.proxy_port}")

    def check_tor_connection(self):
        """Verify that we're connected through Tor"""
        try:
            # Try to connect to the Tor check service
            response = self.session.get("https://check.torproject.org/")

            # Check if the page indicates we're using Tor
            if "Congratulations. This browser is configured to use Tor." in response.text:
                logger.info("Successfully connected through Tor")
                return True
            else:
                logger.warning("Connected to Tor check service, but not using Tor")
                return False

        except Exception as e:
            logger.error(f"Error checking Tor connection: {e}")
            return False

    def random_delay(self):
        """Implement a random delay between requests to be respectful"""
        delay_time = random.uniform(self.delay[0], self.delay[1])
        logger.info(f"Waiting {delay_time:.2f} seconds before next request")
        time.sleep(delay_time)

    def clean_text(self, text):
        """Basic text cleaning"""
        if not text:
            return ""

        # Remove HTML tags
        clean = re.sub(r'<.*?>', ' ', text)
        # Remove URLs
        clean = re.sub(r'http\S+', '', clean)
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def tokenize_text(self, text):
        """Tokenize text for basic analysis"""
        tokens = word_tokenize(text.lower())
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        return tokens

    def save_data(self, data, filename):
        """Save scraped data to JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Data saved to {filepath}")

    def extract_thread_info(self, soup):
        """Extract thread title and other information"""
        thread_title = "Unknown Thread"

        # Try different selectors for the thread title
        title_elem = soup.select_one('.maintitle')
        if title_elem:
            thread_title = title_elem.text.strip()
        else:
            # Fallback to the page title
            if soup.title:
                thread_title = soup.title.text.strip()

        return {
            "title": thread_title,
            "url": soup.url if hasattr(soup, 'url') else "Unknown URL"
        }

    def get_total_pages(self, soup):
        """Extract the total number of pages in the thread"""
        try:
            # Look for pagination information
            pagination = soup.select('.gensmall')
            for page_info in pagination:
                if 'Pages' in page_info.text:
                    # Extract the highest page number
                    page_numbers = [int(num) for num in re.findall(r'\d+', page_info.text)]
                    if page_numbers:
                        return max(page_numbers)
        except Exception as e:
            logger.error(f"Error extracting pagination info: {e}")

        # Default to 1 if we can't determine the number of pages
        return 1

    def construct_page_url(self, base_url, page_num):
        """Construct URL for a specific page in the thread"""
        # Look for existing page parameter
        if '&start=' in base_url or '?start=' in base_url:
            # Replace the existing page parameter
            return re.sub(r'([?&])start=\d+', f'\\1start={15 * (page_num - 1)}', base_url)
        elif '?' in base_url:
            # Add page parameter to existing query string
            return f"{base_url}&start={15 * (page_num - 1)}"
        else:
            # Add page parameter as new query string
            return f"{base_url}?start={15 * (page_num - 1)}"

    def extract_posts(self, soup, thread_url, thread_title):
        """Extract posts from the thread page"""
        posts_data = []

        # Look for post containers - try multiple selector patterns
        post_containers = soup.select(".post.classic")

        # If no posts found with that selector, try alternative selectors
        if not post_containers:
            post_containers = soup.select('[id^="post_"]')

        # If still no posts, try another common pattern
        if not post_containers:
            post_containers = soup.select('div[class*="post"][class*="classic"]')

        # Add more fallbacks if needed
        if not post_containers:
            # Try to find posts via the post content structure
            post_contents = soup.select(".post_content")
            if post_contents:
                # Try to get their parent elements which should be posts
                post_containers = []
                for content in post_contents:
                    parent = content.parent
                    if parent and 'post' in parent.get('class', []):
                        post_containers.append(parent)

        logger.info(f"Found {len(post_containers)} posts on this page")

        for post in post_containers:
            try:
                # Extract post ID
                post_id = post.get('id', '')

                # Extract author information
                author_elem = post.select_one(".post_author strong")
                author = author_elem.text.strip() if author_elem else "Anonymous"

                # Extract timestamp
                timestamp_elem = post.select_one(".post_date")
                raw_timestamp = timestamp_elem.text.strip() if timestamp_elem else "Unknown"
                # Clean up timestamp
                timestamp = re.sub(r'\s+', ' ', raw_timestamp)

                # Extract post content - try different selectors based on the HTML structure
                # First try by direct class
                content_elem = post.select_one(".post_body_scaleimages")

                # If that doesn't work, look for it by ID pattern
                if not content_elem:
                    # Look for divs with IDs matching the pattern pid_*
                    pid_elems = post.select('[id^="pid_"]')
                    if pid_elems:
                        content_elem = pid_elems[0]

                # If still not found, try getting via the post_content container
                if not content_elem:
                    post_content = post.select_one(".post_content")
                    if post_content:
                        content_elem = post_content.select_one('[id^="pid_"]')

                content = ""
                if content_elem:
                    # Get the text content including all nested elements
                    content = content_elem.get_text(separator=' ', strip=True)

                # If still empty, try to get all text within post_content as a fallback
                if not content:
                    post_content = post.select_one(".post_content")
                    if post_content:
                        content = post_content.get_text(separator=' ', strip=True)

                clean_content = self.clean_text(content)

                # Create post data structure
                post_data = {
                    "source": "darkweb",
                    "forum": "SuprBay",
                    "thread_url": thread_url,
                    "thread_title": thread_title,
                    "post_id": post_id,
                    "username": author,
                    "timestamp": timestamp,
                    "content": clean_content,
                    "tokens": self.tokenize_text(clean_content),
                    "scrape_time": datetime.now().isoformat()
                }

                posts_data.append(post_data)

            except Exception as e:
                logger.error(f"Error processing post: {e}")

        return posts_data

    def scrape_thread(self, thread_url, max_pages=None):
        """
        Scrape a SuprBay thread

        Args:
            thread_url (str): URL of the thread to scrape
            max_pages (int, optional): Maximum number of pages to scrape

        Returns:
            list: Collected posts data
        """
        # Check Tor connection
        if not self.check_tor_connection():
            logger.error("Not connected through Tor. Aborting.")
            return []

        all_posts = []

        try:
            # Get the first page to extract thread info and pagination
            logger.info(f"Accessing thread: {thread_url}")
            response = self.session.get(thread_url)
            html_content = response.text

            # Save HTML for debugging if enabled
            if self.debug:
                debug_file = os.path.join(self.debug_dir, f"thread_page_1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                logger.info(f"Saved HTML to {debug_file} for debugging")

            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract thread information
            thread_info = self.extract_thread_info(soup)
            thread_title = thread_info["title"]
            logger.info(f"Thread title: {thread_title}")

            # Get total number of pages
            total_pages = self.get_total_pages(soup)
            logger.info(f"Thread has {total_pages} pages")

            # Limit the number of pages if specified
            if max_pages and max_pages < total_pages:
                total_pages = max_pages
                logger.info(f"Limiting to {max_pages} pages")

            # Process the first page (already loaded)
            logger.info("Processing page 1")
            posts = self.extract_posts(soup, thread_url, thread_title)
            all_posts.extend(posts)

            # Process subsequent pages
            for page_num in range(2, total_pages + 1):
                self.random_delay()
                page_url = self.construct_page_url(thread_url, page_num)
                logger.info(f"Processing page {page_num}: {page_url}")

                try:
                    response = self.session.get(page_url)
                    soup = BeautifulSoup(response.text, 'html.parser')

                    posts = self.extract_posts(soup, thread_url, thread_title)
                    all_posts.extend(posts)

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")

            # Save the data
            thread_id = re.search(r'Thread-([^?&]+)', thread_url)
            thread_id = thread_id.group(1) if thread_id else "unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"suprbay_thread_{thread_id}_{timestamp}.json"
            self.save_data(all_posts, filename)

            logger.info(f"Successfully scraped {len(all_posts)} posts from thread: {thread_title}")
            return all_posts

        except Exception as e:
            logger.error(f"Error scraping thread {thread_url}: {e}")
            return []

def scrape_from_list(thread_list_file, output_dir="data/darkweb", proxy_port=9150, max_pages=None):
    """
    Scrape multiple threads from a list file

    Args:
        thread_list_file (str): Path to a text file containing thread URLs (one per line)
        output_dir (str): Directory to store scraped data
        proxy_port (int): Tor SOCKS proxy port
        max_pages (int, optional): Maximum number of pages to scrape per thread
    """
    try:
        # Read the thread URLs from the file
        with open(thread_list_file, 'r') as f:
            thread_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not thread_urls:
            logger.error("No thread URLs found in the file")
            return

        logger.info(f"Found {len(thread_urls)} thread URLs to scrape")

        # Initialize the scraper
        scraper = SuprBayScraper(output_dir=output_dir, proxy_port=proxy_port)

        # Scrape each thread
        for i, url in enumerate(thread_urls, 1):
            logger.info(f"Scraping thread {i}/{len(thread_urls)}: {url}")
            scraper.scrape_thread(url, max_pages=max_pages)

            # Pause between threads
            if i < len(thread_urls):
                delay = random.uniform(15, 30)
                logger.info(f"Waiting {delay:.2f} seconds before the next thread")
                time.sleep(delay)

        logger.info("All threads have been scraped successfully")

    except Exception as e:
        logger.error(f"Error reading thread list file: {e}")


def main():
    """Command line interface for the scraper"""
    parser = argparse.ArgumentParser(description="SuprBay Forum Thread Scraper")

    parser.add_argument("--thread-url", help="URL of a specific thread to scrape")
    parser.add_argument("--thread-list", help="Path to a text file containing thread URLs to scrape")
    parser.add_argument("--output-dir", default="data/darkweb", help="Directory to store scraped data")
    parser.add_argument("--tor-port", type=int, default=9150, help="Tor SOCKS proxy port")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to scrape per thread")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save HTML for inspection")
    parser.add_argument("--delay", type=int, default=10, help="Base delay between requests in seconds")

    args = parser.parse_args()

    if not args.thread_url and not args.thread_list:
        parser.error("Either --thread-url or --thread-list must be specified")

    # Adjust delay based on command line argument
    delay_range = (args.delay, args.delay * 2)

    if args.thread_url:
        # Scrape a single thread
        scraper = SuprBayScraper(
            output_dir=args.output_dir,
            proxy_port=args.tor_port,
            delay=delay_range,
            debug=args.debug
        )
        scraper.scrape_thread(args.thread_url, max_pages=args.max_pages)

    if args.thread_list:
        # Scrape multiple threads from a list
        # Update the scrape_from_list function to pass debug parameter
        scraper = SuprBayScraper(
            output_dir=args.output_dir,
            proxy_port=args.tor_port,
            delay=delay_range,
            debug=args.debug
        )

        # Read the thread URLs from the file
        with open(args.thread_list, 'r') as f:
            thread_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not thread_urls:
            logger.error("No thread URLs found in the file")
            return

        logger.info(f"Found {len(thread_urls)} thread URLs to scrape")

        # Scrape each thread
        for i, url in enumerate(thread_urls, 1):
            logger.info(f"Scraping thread {i}/{len(thread_urls)}: {url}")
            scraper.scrape_thread(url, max_pages=args.max_pages)

            # Pause between threads
            if i < len(thread_urls):
                delay = random.uniform(15, 30)
                logger.info(f"Waiting {delay:.2f} seconds before the next thread")
                time.sleep(delay)


if __name__ == "__main__":
    main()