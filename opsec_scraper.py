"""
OpSec Discourse Analysis Scraper
--------------------------------
A comprehensive tool for collecting OpSec discussions from both Dark Web and Surface Web forums.
This script supports:
- Dark Web scraping via Tor
- Surface Web scraping (including Reddit)
- Basic text processing and data storage
"""

import os
import time
import json
import random
import argparse
import logging
from datetime import datetime
from urllib.parse import urlparse

# Web scraping libraries
import requests
from bs4 import BeautifulSoup

# For Tor connectivity
import socks
import socket

# NLP libraries for basic text processing
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("opsec_scraper")

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class OpSecScraper:
    """Base scraper class with common functionality"""

    def __init__(self, output_dir="data", delay=(5, 15)):
        """
        Initialize the scraper

        Args:
            output_dir (str): Directory to store scraped data
            delay (tuple): Random delay range between requests (min, max) in seconds
        """
        self.output_dir = output_dir
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def random_delay(self):
        """Implement a random delay between requests to be respectful"""
        delay_time = random.uniform(self.delay[0], self.delay[1])
        logger.info(f"Waiting {delay_time:.2f} seconds before next request")
        time.sleep(delay_time)

    def save_data(self, data, filename):
        """Save scraped data to JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Data saved to {filepath}")

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

    def extract_metadata(self, soup, post_timestamp_selector, username_selector):
        """Extract metadata from a post (to be implemented by child classes)"""
        pass

    def parse_content(self, soup, content_selector):
        """Parse content from a post (to be implemented by child classes)"""
        pass

    def scrape_forum(self, url, num_pages):
        """Scrape forum posts (to be implemented by child classes)"""
        pass


class DarkWebScraper(OpSecScraper):
    """Scraper for Dark Web forums using Tor"""

    def __init__(self, output_dir="data/darkweb", proxy_port=9050, delay=(10, 20)):
        """
        Initialize the Dark Web scraper

        Args:
            output_dir (str): Directory to store scraped data
            proxy_port (int): Tor SOCKS proxy port
            delay (tuple): Random delay range between requests (min, max) in seconds
        """
        super().__init__(output_dir, delay)
        self.proxy_port = proxy_port
        self.setup_tor_connection()

    def setup_tor_connection(self):
        """Configure the session to route through Tor"""
        # Set up the SOCKS proxy for Tor
        socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", self.proxy_port)
        socket.socket = socks.socksocket

        # Configure requests to use the SOCKS proxy
        self.session.proxies = {
            'http': 'socks5h://127.0.0.1:' + str(self.proxy_port),
            'https': 'socks5h://127.0.0.1:' + str(self.proxy_port)
        }

        logger.info("Tor connection established")

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

    def scrape_forum(self, forum_url, num_pages=5,
                     thread_selector=".thread",
                     thread_link_selector="a.thread-title",
                     post_selector=".post",
                     username_selector=".username",
                     timestamp_selector=".post-time",
                     content_selector=".post-content"):
        """
        Scrape a Dark Web forum

        Args:
            forum_url (str): Base URL of the forum
            num_pages (int): Number of pages to scrape
            thread_selector (str): CSS selector for thread elements
            thread_link_selector (str): CSS selector for thread links
            post_selector (str): CSS selector for post elements
            username_selector (str): CSS selector for username
            timestamp_selector (str): CSS selector for post timestamp
            content_selector (str): CSS selector for post content

        Returns:
            list: Collected posts data
        """
        # Check Tor connection
        if not self.check_tor_connection():
            logger.error("Not connected through Tor. Aborting.")
            return []

        collected_data = []

        try:
            # First, get forum index to find thread links
            logger.info(f"Accessing forum index: {forum_url}")
            response = self.session.get(forum_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract thread links
            thread_elements = soup.select(thread_selector)
            thread_links = []

            for thread in thread_elements:
                link_element = thread.select_one(thread_link_selector)
                if link_element and link_element.has_attr('href'):
                    thread_url = link_element['href']
                    # Handle relative URLs
                    if not thread_url.startswith('http'):
                        parsed_base = urlparse(forum_url)
                        thread_url = f"{parsed_base.scheme}://{parsed_base.netloc}{thread_url}"
                    thread_links.append(thread_url)

            logger.info(f"Found {len(thread_links)} threads")

            # Limit the number of threads to scrape
            thread_links = thread_links[:min(len(thread_links), num_pages)]

            # Scrape each thread
            for thread_url in thread_links:
                logger.info(f"Scraping thread: {thread_url}")
                self.random_delay()

                try:
                    thread_response = self.session.get(thread_url)
                    thread_soup = BeautifulSoup(thread_response.text, 'html.parser')

                    # Extract thread title
                    thread_title = thread_soup.title.text if thread_soup.title else "Unknown Thread"
                    thread_title = self.clean_text(thread_title)

                    # Extract posts
                    posts = thread_soup.select(post_selector)

                    for post in posts:
                        try:
                            # Extract post data
                            username_elem = post.select_one(username_selector)
                            username = username_elem.text.strip() if username_elem else "Anonymous"

                            timestamp_elem = post.select_one(timestamp_selector)
                            timestamp = timestamp_elem.text.strip() if timestamp_elem else "Unknown"

                            content_elem = post.select_one(content_selector)
                            content = content_elem.text.strip() if content_elem else ""
                            clean_content = self.clean_text(content)

                            # Create post data structure
                            post_data = {
                                "source": "darkweb",
                                "forum": urlparse(forum_url).netloc,
                                "thread_url": thread_url,
                                "thread_title": thread_title,
                                "username": username,
                                "timestamp": timestamp,
                                "content": clean_content,
                                "tokens": self.tokenize_text(clean_content),
                                "scrape_time": datetime.now().isoformat()
                            }

                            collected_data.append(post_data)

                        except Exception as e:
                            logger.error(f"Error processing post: {e}")

                except Exception as e:
                    logger.error(f"Error scraping thread {thread_url}: {e}")

            # Save the collected data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"darkweb_{urlparse(forum_url).netloc}_{timestamp}.json"
            self.save_data(collected_data, filename)

            return collected_data

        except Exception as e:
            logger.error(f"Error scraping forum {forum_url}: {e}")
            return []


class SurfaceWebScraper(OpSecScraper):
    """Scraper for Surface Web forums"""

    def __init__(self, output_dir="data/surfaceweb", delay=(3, 8)):
        """
        Initialize the Surface Web scraper

        Args:
            output_dir (str): Directory to store scraped data
            delay (tuple): Random delay range between requests (min, max) in seconds
        """
        super().__init__(output_dir, delay)

    def scrape_forum(self, forum_url, num_pages=5,
                     thread_selector=".thread",
                     thread_link_selector="a.thread-title",
                     pagination_selector=".pagination a",
                     post_selector=".post",
                     username_selector=".username",
                     timestamp_selector=".post-time",
                     content_selector=".post-content"):
        """
        Scrape a Surface Web forum

        Args:
            forum_url (str): Base URL of the forum
            num_pages (int): Number of pages to scrape
            thread_selector (str): CSS selector for thread elements
            thread_link_selector (str): CSS selector for thread links
            pagination_selector (str): CSS selector for pagination links
            post_selector (str): CSS selector for post elements
            username_selector (str): CSS selector for username
            timestamp_selector (str): CSS selector for post timestamp
            content_selector (str): CSS selector for post content

        Returns:
            list: Collected posts data
        """
        collected_data = []

        try:
            logger.info(f"Starting to scrape {forum_url}")

            # Get the forum index
            response = self.session.get(forum_url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Collect pagination links if available
            page_links = [forum_url]
            pagination = soup.select(pagination_selector)

            for i in range(1, min(num_pages, len(pagination) + 1)):
                if i < len(pagination):
                    page_url = pagination[i].get('href')
                    # Handle relative URLs
                    if page_url and not page_url.startswith('http'):
                        parsed_base = urlparse(forum_url)
                        page_url = f"{parsed_base.scheme}://{parsed_base.netloc}{page_url}"
                    if page_url:
                        page_links.append(page_url)

            # Limit to num_pages
            page_links = page_links[:num_pages]

            # Process each page
            for page_url in page_links:
                logger.info(f"Processing page: {page_url}")
                self.random_delay()

                try:
                    page_response = self.session.get(page_url)
                    page_soup = BeautifulSoup(page_response.text, 'html.parser')

                    # Extract thread links
                    thread_elements = page_soup.select(thread_selector)
                    thread_links = []

                    for thread in thread_elements:
                        link_element = thread.select_one(thread_link_selector)
                        if link_element and link_element.has_attr('href'):
                            thread_url = link_element['href']
                            # Handle relative URLs
                            if not thread_url.startswith('http'):
                                parsed_base = urlparse(forum_url)
                                thread_url = f"{parsed_base.scheme}://{parsed_base.netloc}{thread_url}"
                            thread_links.append(thread_url)

                    # Process each thread
                    for thread_url in thread_links:
                        logger.info(f"Scraping thread: {thread_url}")
                        self.random_delay()

                        try:
                            thread_response = self.session.get(thread_url)
                            thread_soup = BeautifulSoup(thread_response.text, 'html.parser')

                            # Extract thread title
                            thread_title = thread_soup.title.text if thread_soup.title else "Unknown Thread"
                            thread_title = self.clean_text(thread_title)

                            # Extract posts
                            posts = thread_soup.select(post_selector)

                            for post in posts:
                                try:
                                    # Extract post data
                                    username_elem = post.select_one(username_selector)
                                    username = username_elem.text.strip() if username_elem else "Anonymous"

                                    timestamp_elem = post.select_one(timestamp_selector)
                                    timestamp = timestamp_elem.text.strip() if timestamp_elem else "Unknown"

                                    content_elem = post.select_one(content_selector)
                                    content = content_elem.text.strip() if content_elem else ""
                                    clean_content = self.clean_text(content)

                                    # Create post data structure
                                    post_data = {
                                        "source": "surfaceweb",
                                        "forum": urlparse(forum_url).netloc,
                                        "thread_url": thread_url,
                                        "thread_title": thread_title,
                                        "username": username,
                                        "timestamp": timestamp,
                                        "content": clean_content,
                                        "tokens": self.tokenize_text(clean_content),
                                        "scrape_time": datetime.now().isoformat()
                                    }

                                    collected_data.append(post_data)

                                except Exception as e:
                                    logger.error(f"Error processing post: {e}")

                        except Exception as e:
                            logger.error(f"Error scraping thread {thread_url}: {e}")

                except Exception as e:
                    logger.error(f"Error processing page {page_url}: {e}")

            # Save the collected data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"surfaceweb_{urlparse(forum_url).netloc}_{timestamp}.json"
            self.save_data(collected_data, filename)

            return collected_data

        except Exception as e:
            logger.error(f"Error scraping forum {forum_url}: {e}")
            return []


class RedditScraper(SurfaceWebScraper):
    """Specialized scraper for Reddit using PRAW or Reddit API"""

    def __init__(self, output_dir="data/reddit",
                 client_id=None, client_secret=None, user_agent=None,
                 delay=(3, 8)):
        """
        Initialize the Reddit scraper

        Args:
            output_dir (str): Directory to store scraped data
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): Reddit API user agent
            delay (tuple): Random delay range between requests (min, max) in seconds
        """
        super().__init__(output_dir, delay)
        self.client_id = client_id
        self.client_secret = client_secret

        if user_agent is None:
            self.user_agent = "OpSecDiscourseAnalysis/1.0 (Academic Research Project)"
        else:
            self.user_agent = user_agent

        # Note: In a full implementation, you would use PRAW (Python Reddit API Wrapper)
        # But for simplicity and to avoid additional dependencies, we'll use the public JSON API

    def scrape_subreddit(self, subreddit_name, limit=25, time_filter="month"):
        """
        Scrape posts and comments from a subreddit

        Args:
            subreddit_name (str): Name of the subreddit (without r/)
            limit (int): Maximum number of posts to scrape
            time_filter (str): Time filter for top posts (hour, day, week, month, year, all)

        Returns:
            list: Collected posts and comments data
        """
        collected_data = []

        try:
            # Construct the URL for the subreddit's top posts in JSON format
            url = f"https://www.reddit.com/r/{subreddit_name}/top/.json?t={time_filter}&limit={limit}"
            logger.info(f"Scraping subreddit: r/{subreddit_name}")

            # Get the listing of posts
            response = self.session.get(url, headers={'User-Agent': self.user_agent})
            data = response.json()

            # Process each post
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})

                # Extract post information
                post_id = post_data.get('id', 'unknown')
                post_title = post_data.get('title', '')
                post_content = post_data.get('selftext', '')
                post_url = post_data.get('url', '')
                post_created = post_data.get('created_utc', 0)
                author = post_data.get('author', 'deleted')

                # Convert timestamp
                post_timestamp = datetime.fromtimestamp(post_created).isoformat()

                # Clean the content
                clean_content = self.clean_text(post_content)

                # Create post data structure
                post_item = {
                    "source": "reddit",
                    "forum": f"r/{subreddit_name}",
                    "post_id": post_id,
                    "thread_url": f"https://www.reddit.com{post_data.get('permalink', '')}",
                    "thread_title": post_title,
                    "username": author,
                    "timestamp": post_timestamp,
                    "content": clean_content,
                    "tokens": self.tokenize_text(clean_content),
                    "is_post": True,
                    "scrape_time": datetime.now().isoformat()
                }

                collected_data.append(post_item)

                # Get comments for this post
                self.random_delay()
                comments_url = f"https://www.reddit.com{post_data.get('permalink', '')}.json"
                comments_response = self.session.get(comments_url, headers={'User-Agent': self.user_agent})

                if comments_response.status_code == 200:
                    comments_data = comments_response.json()

                    # The second element in the array contains the comments
                    if len(comments_data) > 1:
                        comments_listing = comments_data[1].get('data', {}).get('children', [])

                        # Process each top-level comment
                        for comment in comments_listing:
                            if comment.get('kind') == 't1':  # t1 is the prefix for comments
                                comment_data = comment.get('data', {})

                                # Extract comment information
                                comment_id = comment_data.get('id', 'unknown')
                                comment_body = comment_data.get('body', '')
                                comment_created = comment_data.get('created_utc', 0)
                                comment_author = comment_data.get('author', 'deleted')

                                # Convert timestamp
                                comment_timestamp = datetime.fromtimestamp(comment_created).isoformat()

                                # Clean the content
                                clean_comment = self.clean_text(comment_body)

                                # Create comment data structure
                                comment_item = {
                                    "source": "reddit",
                                    "forum": f"r/{subreddit_name}",
                                    "post_id": post_id,
                                    "comment_id": comment_id,
                                    "thread_url": f"https://www.reddit.com{post_data.get('permalink', '')}",
                                    "thread_title": post_title,
                                    "username": comment_author,
                                    "timestamp": comment_timestamp,
                                    "content": clean_comment,
                                    "tokens": self.tokenize_text(clean_comment),
                                    "is_post": False,
                                    "scrape_time": datetime.now().isoformat()
                                }

                                collected_data.append(comment_item)

            # Save the collected data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_{subreddit_name}_{timestamp}.json"
            self.save_data(collected_data, filename)

            return collected_data

        except Exception as e:
            logger.error(f"Error scraping subreddit r/{subreddit_name}: {e}")
            return []


def main():
    """Main function to run the script with command line arguments"""
    parser = argparse.ArgumentParser(description="OpSec Discourse Analysis Scraper")

    # General arguments
    parser.add_argument("--output-dir", default="data", help="Directory to store scraped data")

    # Dark Web arguments
    parser.add_argument("--darkweb", action="store_true", help="Scrape Dark Web forums")
    parser.add_argument("--tor-port", type=int, default=9050, help="Tor SOCKS proxy port")
    parser.add_argument("--darkweb-url", help="URL of the Dark Web forum to scrape")

    # Surface Web arguments
    parser.add_argument("--surfaceweb", action="store_true", help="Scrape Surface Web forums")
    parser.add_argument("--surfaceweb-url", help="URL of the Surface Web forum to scrape")

    # Reddit arguments
    parser.add_argument("--reddit", action="store_true", help="Scrape Reddit")
    parser.add_argument("--subreddit", help="Name of the subreddit to scrape (without r/)")
    parser.add_argument("--reddit-limit", type=int, default=25, help="Maximum number of Reddit posts to scrape")

    # Scraping parameters
    parser.add_argument("--num-pages", type=int, default=5, help="Number of pages to scrape")

    args = parser.parse_args()

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Scrape Dark Web forums
    if args.darkweb:
        if not args.darkweb_url:
            logger.error("Dark Web URL is required for scraping Dark Web forums")
        else:
            dark_web_scraper = DarkWebScraper(
                output_dir=os.path.join(args.output_dir, "darkweb"),
                proxy_port=args.tor_port
            )
            dark_web_scraper.scrape_forum(args.darkweb_url, num_pages=args.num_pages)

    # Scrape Surface Web forums
    if args.surfaceweb:
        if not args.surfaceweb_url:
            logger.error("Surface Web URL is required for scraping Surface Web forums")
        else:
            surface_web_scraper = SurfaceWebScraper(
                output_dir=os.path.join(args.output_dir, "surfaceweb")
            )
            surface_web_scraper.scrape_forum(args.surfaceweb_url, num_pages=args.num_pages)

    # Scrape Reddit
    if args.reddit:
        if not args.subreddit:
            logger.error("Subreddit name is required for scraping Reddit")
        else:
            reddit_scraper = RedditScraper(
                output_dir=os.path.join(args.output_dir, "reddit")
            )
            reddit_scraper.scrape_subreddit(args.subreddit, limit=args.reddit_limit)


if __name__ == "__main__":
    main()