import os
import time
import json
import random
import argparse
import logging
import re
from datetime import datetime
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
import socks
import socket
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Logs in case of any errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("suprbay_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("suprbay_scraper")

# NLTK resources for text processing
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class SuprBayScraper:
    """Scraper for SuprBay forum"""

    def __init__(self, output_dir="data/darkweb", proxy_port=9150, delay=(8, 15), debug=False):
        """
        Initialize the SuprBay scraper

        Args:
            output_dir (str): Directory to store scraped data
            proxy_port (int): Tor SOCKS proxy port - 9150 for Tor Browser
            delay (tuple): Random delay range between requests (min, max) in seconds
            debug (bool): Whether to save HTML for debugging purposes
        """
        self.output_dir = output_dir
        self.delay = delay
        self.proxy_port = proxy_port
        self.debug = debug
        self.base_url = "http://suprbaydvdcaynfo4dgdzgxb4zuso7rftlil5yg5kqjefnw4wq4ulcad.onion" # SuprBay Dark Web Forum URL, always need to check if this is still active

        # Debug directory if needed
        if self.debug:
            self.debug_dir = os.path.join(output_dir, "debug")
            os.makedirs(self.debug_dir, exist_ok=True)
            
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Tor connection
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
        """Random delay between requests"""
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
        # Removing stopwords
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

        # Trying different selectors for the thread title
        title_elem = soup.select_one('.cattitle')
        if title_elem:
            thread_title = title_elem.text.strip()
        elif soup.select_one('.maintitle'):
            thread_title = soup.select_one('.maintitle').text.strip()
        else:
            # Fallback to the page title
            if soup.title:
                thread_title = soup.title.text.strip()

        return {
            "title": thread_title,
            "url": soup.url if hasattr(soup, 'url') else "Unknown URL"
        }

    def get_total_pages(self, soup, is_forum_section=False):
        """
        Extracting the total number of pages in the thread or forum section
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            is_forum_section (bool): Whether this is a forum section page or thread page
        """
        try:
            # First try to find the pagination info from the navigation bar
            nav = soup.select_one('.pagenav')
            if nav:
                # Looking specifically for the last page number in pagination
                last_page = nav.select('a.navpage')[-1] if nav.select('a.navpage') else None
                if last_page and last_page.text.strip().isdigit():
                    return int(last_page.text.strip())
                
                # If no explicit page numbers, checking the last start parameter
                last_link = nav.select('a[href*="start="]')[-1] if nav.select('a[href*="start="]') else None
                if last_link and last_link.has_attr('href'):
                    start_match = re.search(r'start=(\d+)', last_link['href'])
                    if start_match:
                        start = int(start_match.group(1))
                        return (start // 15) + 1
            
            # Looking for post count and calculating pages (on average there are15 posts per page)
            post_count_elem = soup.select_one('.thread-stats')
            if post_count_elem:
                post_count_match = re.search(r'Replies:\s*(\d+)', post_count_elem.text)
                if post_count_match:
                    # Adding 1 to replies for the original post
                    post_count = int(post_count_match.group(1)) + 1
                    return max(1, (post_count + 14) // 15)  # Round up division by 15
            
            # If we can't find any pagination info, checking if there's a "Next" link
            next_link = soup.find('a', string=lambda s: s and ('Next' in s or '»' in s))
            if next_link:
                return 2  # At least 2 pages if there's a Next link
            
            # Default to 1 if we can't determine the number of pages
            return 1
            
        except Exception as e:
            logger.error(f"Error extracting pagination info: {e}")
            return 1

    def construct_page_url(self, base_url, page_num, is_forum_section=False):
        """
        Constructing URL for a specific page in the thread or forum section
        
        Args:
            base_url (str): Base URL
            page_num (int): Page number
            is_forum_section (bool): Whether this is a forum section page or thread page
        """
        if is_forum_section:
            # For forum section pages
            if page_num == 1:
                return base_url
                
            # Checking for existing pagination format
            if '&page=' in base_url or '?page=' in base_url:
                # Replacing the existing page parameter
                return re.sub(r'([?&])page=\d+', f'\\1page={page_num}', base_url)
            elif '?' in base_url:
                # Add page parameter to existing query string
                return f"{base_url}&page={page_num}"
            else:
                # Add page parameter as new query string
                return f"{base_url}?page={page_num}"
        else:
            # For thread pages
            if page_num == 1:
                return base_url
                
            # SuprBay uses different formats for thread pagination
            if '&start=' in base_url or '?start=' in base_url:
                # Replacing the existing start parameter
                return re.sub(r'([?&])start=\d+', f'\\1start={15 * (page_num - 1)}', base_url)
            elif '?' in base_url:
                # Adding start parameter to existing query string
                return f"{base_url}&start={15 * (page_num - 1)}"
            else:
                # Adding start parameter as new query string
                return f"{base_url}?start={15 * (page_num - 1)}"

    def extract_posts(self, soup, thread_url, thread_title):
        """Extract posts from the thread page"""
        posts_data = []

        # Looking for post containers - trying multiple selector patterns
        post_containers = soup.select("table.tborder tr:has(.post_body_scaleimages)")
        
        if not post_containers:
            post_containers = soup.select(".post.classic, [id^='post_'], div[class*='post'][class*='classic']")
        
        if not post_containers:
            post_containers = soup.select("tr:has(.post_content)")

        logger.info(f"Found {len(post_containers)} posts on this page")

        for post in post_containers:
            try:
                # Extracting post ID
                post_id = post.get('id', '')
                if not post_id:
                    # Trying to find id in nested elements
                    id_elem = post.select_one('[id^="post_"], [id^="pid_"]')
                    if id_elem:
                        post_id = id_elem.get('id', '')

                # Extracting author information - trying various selectors
                author = "Anonymous"
                author_elems = [
                    post.select_one(".post_author strong"),
                    post.select_one(".username"),
                    post.select_one(".name"),
                    post.select_one("td.profile strong"),
                    post.select_one("td.profile a")
                ]
                
                for author_elem in author_elems:
                    if author_elem:
                        author_text = author_elem.text.strip()
                        if author_text:
                            author = author_text
                            break

                # Extracting timestamp - trying various selectors
                timestamp = "Unknown"
                timestamp_elems = [
                    post.select_one(".post_date"),
                    post.select_one(".postdate"),
                    post.select_one(".post-time")
                ]
                
                for timestamp_elem in timestamp_elems:
                    if timestamp_elem:
                        timestamp_text = timestamp_elem.text.strip()
                        if timestamp_text:
                            timestamp = re.sub(r'\s+', ' ', timestamp_text)
                            break

                # Extracting post content - trying different methods
                content = ""
                
                # Method 1: Direct class selector
                content_elem = post.select_one(".post_body_scaleimages")
                
                # Method 2: ID pattern
                if not content_elem or not content_elem.text.strip():
                    pid_elems = post.select('[id^="pid_"]')
                    if pid_elems:
                        content_elem = pid_elems[0]
                
                # Method 3: Post content container
                if not content_elem or not content_elem.text.strip():
                    post_content = post.select_one(".post_content")
                    if post_content:
                        content_elem = post_content.select_one('[id^="pid_"]')
                        if not content_elem:
                            content_elem = post_content  # Use post_content itself
                
                # Extracting text from the found element
                if content_elem:
                    content = content_elem.get_text(separator=' ', strip=True)
                
                # Last resort: Trying to get any text from the post
                if not content:
                    # Excluding author and timestamp areas
                    for text_elem in post.find_all(text=True):
                        if text_elem.parent and not text_elem.parent.name == 'script' and not text_elem.parent.name == 'style':
                            if text_elem.strip() and text_elem.strip() not in [author, timestamp]:
                                content += text_elem.strip() + " "

                clean_content = self.clean_text(content)

                # Creating post data structure output
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

    def _process_thread_link(self, link, thread_title):
        """
        Processing a thread link and returning thread data if valid
        
        Args:
            link (BeautifulSoup element): The link element
            thread_title (str): The thread title
            
        Returns:
            dict: Thread data if valid, None otherwise
        """
        if not link.has_attr('href') or not thread_title:
            return None
        
        # Skipping FAQ threads
        if any(keyword in thread_title.lower() for keyword in ['faq', 'frequently asked', 'how to', 'guide']):
            logger.info(f"Skipping FAQ thread: {thread_title}")
            return None
        
        thread_url = link['href']
        
        # Handling relative URLs
        if not thread_url.startswith('http'):
            # Checking if it starts with a slash
            if thread_url.startswith('/'):
                thread_url = f"{self.base_url}{thread_url}"
            else:
                thread_url = f"{self.base_url}/{thread_url}"
        
        return {
            'url': thread_url,
            'title': thread_title
        }

    def extract_thread_links(self, soup, forum_url):
        """
        Extracting thread links from a forum section page
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            forum_url (str): Forum section URL
        
        Returns:
            list: List of thread URLs with titles
        """
        thread_links = []
        
        # Based on the HTML structure from the screenshot, adjusting the selectors
        # SuprBay has threads in table rows with specific IDs!!!
        thread_rows = soup.select("tr[id^='thread_']")
        
        if not thread_rows:
            # Trying another common format
            thread_rows = soup.select("tr:has(a.topictitle), tr:has(td.threadcol)")
        
        for row in thread_rows:
            try:
                # Finding the thread title and link - trying multiple selector patterns
                title_elem = None
                
                # Trying different selectors
                for selector in ["td.threadcol a", "a.topictitle", "a.thread-title", ".thread-link"]:
                    title_elem = row.select_one(selector)
                    if title_elem and title_elem.has_attr('href'):
                        break
                
                if title_elem:
                    thread_data = self._process_thread_link(title_elem, title_elem.text.strip())
                    if thread_data:
                        thread_links.append(thread_data)
                    
            except Exception as e:
                logger.error(f"Error extracting thread link: {e}")
        
        # If we still can't find threads, trying a more general approach
        if not thread_links:
            # Looking for links with "Thread-" in the URL
            for link in soup.select("a[href*='Thread-']"):
                try:
                    thread_data = self._process_thread_link(link, link.text.strip())
                    if thread_data:
                        thread_links.append(thread_data)
                except Exception as e:
                    logger.error(f"Error extracting thread link: {e}")
        
        # Deduplicating thread links
        unique_links = []
        seen_urls = set()
        for thread in thread_links:
            if thread['url'] not in seen_urls:
                seen_urls.add(thread['url'])
                unique_links.append(thread)
        
        logger.info(f"Found {len(unique_links)} thread links on this page (excluding FAQs)")
        return unique_links

    def scrape_thread(self, thread_url, max_pages=None):
        """
        Scraping a SuprBay thread

        Args:
            thread_url (str): URL of the thread to scrape
            max_pages (int, optional): Maximum number of pages to scrape

        Returns:
            list: Collected posts data
        """
        # Checking Tor connection
        if not self.check_tor_connection():
            logger.error("Not connected through Tor. Aborting.")
            return []

        all_posts = []

        try:
            # Getting the first page to extract thread info and pagination
            logger.info(f"Accessing thread: {thread_url}")
            soup = self._process_page_with_debug(thread_url, "thread_page_1")

            # Extracting thread information
            thread_info = self.extract_thread_info(soup)
            thread_title = thread_info["title"]
            logger.info(f"Thread title: {thread_title}")

            # Getting total number of pages
            total_pages = self.get_total_pages(soup)
            logger.info(f"Thread has {total_pages} pages")

            # Limiting the number of pages if specified
            if max_pages and max_pages < total_pages:
                total_pages = max_pages
                logger.info(f"Limiting to {max_pages} pages")

            # Processing the first page (already loaded)
            logger.info("Processing page 1")
            posts = self.extract_posts(soup, thread_url, thread_title)
            all_posts.extend(posts)

            # Processing subsequent pages
            for page_num in range(2, total_pages + 1):
                self.random_delay()
                page_url = self.construct_page_url(thread_url, page_num)
                logger.info(f"Processing page {page_num}: {page_url}")

                try:
                    soup = self._process_page_with_debug(page_url, f"thread_page_{page_num}")
                    posts = self.extract_posts(soup, thread_url, thread_title)
                    all_posts.extend(posts)

                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {e}")

            # Saving the data
            thread_id = re.search(r'Thread-([^?&/]+)', thread_url)
            thread_id = thread_id.group(1) if thread_id else "unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"suprbay_thread_{thread_id}_{timestamp}.json"
            self.save_data(all_posts, filename)

            logger.info(f"Successfully scraped {len(all_posts)} posts from thread: {thread_title}")
            return all_posts

        except Exception as e:
            logger.error(f"Error scraping thread {thread_url}: {e}")
            return []

    def _process_page_with_debug(self, url, debug_prefix=None):
        """
        Processing a page with debug HTML saving
        
        Args:
            url (str): URL to process
            debug_prefix (str): Prefix for debug file name
            
        Returns:
            BeautifulSoup: Parsed HTML content
        """
        response = self.session.get(url)
        html_content = response.text
        
        # Saving HTML for debugging
        if self.debug and debug_prefix:
            debug_file = os.path.join(
                self.debug_dir,
                f"{debug_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Saved HTML to {debug_file} for debugging")
        
        return BeautifulSoup(html_content, 'html.parser')

    def scrape_forum_section(self, forum_url, max_section_pages=None, max_thread_pages=None, thread_limit=None):
        """
        Scraping all threads from a forum section
        
        Args:
            forum_url (str): URL of the forum section
            max_section_pages (int, optional): Maximum number of section pages to scrape
            max_thread_pages (int, optional): Maximum number of pages to scrape per thread
            thread_limit (int, optional): Maximum number of threads to scrape
            
        Returns:
            list: List of thread URLs that were scraped
        """
        # Checking Tor connection
        if not self.check_tor_connection():
            logger.error("Not connected through Tor. Aborting.")
            return []
            
        all_thread_links = []
        scraped_threads = []
        
        try:
            # Accessing the first page of the forum section
            logger.info(f"Accessing forum section: {forum_url}")
            soup = self._process_page_with_debug(forum_url, "forum_section_page_1")
            
            # Getting the forum title
            forum_title = soup.title.text.strip() if soup.title else "SuprBay Forum"
            logger.info(f"Forum section: {forum_title}")
            
            # Getting total number of pages in the forum section
            total_section_pages = self.get_total_pages(soup, is_forum_section=True)
            logger.info(f"Forum section has {total_section_pages} pages")
            
            # Limiting the number of section pages if specified
            if max_section_pages and max_section_pages < total_section_pages:
                total_section_pages = max_section_pages
                logger.info(f"Limiting to {max_section_pages} section pages")
                
            # Extracting thread links from the first page
            thread_links = self.extract_thread_links(soup, forum_url)
            all_thread_links.extend(thread_links)
            
            # Processing subsequent section pages
            for page_num in range(2, total_section_pages + 1):
                self.random_delay()
                page_url = self.construct_page_url(forum_url, page_num, is_forum_section=True)
                logger.info(f"Processing section page {page_num}: {page_url}")
                
                try:
                    soup = self._process_page_with_debug(page_url, f"forum_section_page_{page_num}")
                    thread_links = self.extract_thread_links(soup, forum_url)
                    all_thread_links.extend(thread_links)
                    
                except Exception as e:
                    logger.error(f"Error processing section page {page_num}: {e}")
            
            # Applying thread limit if specified
            if thread_limit and thread_limit < len(all_thread_links):
                logger.info(f"Limiting to {thread_limit} threads (found {len(all_thread_links)})")
                all_thread_links = all_thread_links[:thread_limit]
                
            # Saving the list of thread links for future reference
            thread_links_data = [{
                "url": thread["url"],
                "title": thread["title"]
            } for thread in all_thread_links]
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            links_filename = f"suprbay_forum_thread_links_{timestamp}.json"
            self.save_data(thread_links_data, links_filename)
            
            # Scraping each thread
            for i, thread in enumerate(all_thread_links):
                logger.info(f"Scraping thread {i+1}/{len(all_thread_links)}: {thread['title']}")
                self.random_delay()
                
                try:
                    self.scrape_thread(thread['url'], max_pages=max_thread_pages)
                    scraped_threads.append(thread['url'])
                    
                except Exception as e:
                    logger.error(f"Error scraping thread {thread['url']}: {e}")
                
                # Pausing between threads
                if i < len(all_thread_links) - 1:
                    delay = random.uniform(15, 30)
                    logger.info(f"Waiting {delay:.2f} seconds before the next thread")
                    time.sleep(delay)
            
            logger.info(f"Successfully scraped {len(scraped_threads)} threads from forum section: {forum_title}")
            return scraped_threads
            
        except Exception as e:
            logger.error(f"Error scraping forum section {forum_url}: {e}")
            return []


def scrape_from_list(thread_list_file, output_dir="data/darkweb", proxy_port=9150, delay_range=(8, 15), debug=False, max_pages=None):
    """
    Scraping multiple threads from a list file

    Args:
        thread_list_file (str): Path to a text file containing thread URLs (one per line)
        output_dir (str): Directory to store scraped data
        proxy_port (int): Tor SOCKS proxy port
        delay_range (tuple): Delay range between requests
        debug (bool): Whether to save HTML for debugging
        max_pages (int, optional): Maximum number of pages to scrape per thread
    """
    try:
        # Reading the thread URLs from the file
        with open(thread_list_file, 'r') as f:
            thread_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not thread_urls:
            logger.error("No thread URLs found in the file")
            return

        logger.info(f"Found {len(thread_urls)} thread URLs to scrape")

        # Initializing the scraper
        scraper = SuprBayScraper(
            output_dir=output_dir, 
            proxy_port=proxy_port, 
            delay=delay_range,
            debug=debug
        )

        # Scraping each thread
        for i, url in enumerate(thread_urls, 1):
            logger.info(f"Scraping thread {i}/{len(thread_urls)}: {url}")
            scraper.scrape_thread(url, max_pages=max_pages)

            # Pausing between threads
            if i < len(thread_urls):
                delay = random.uniform(15, 30)
                logger.info(f"Waiting {delay:.2f} seconds before the next thread")
                time.sleep(delay)

        logger.info("All threads have been scraped successfully")

    except Exception as e:
        logger.error(f"Error reading thread list file: {e}")


def main():
    """Command line interface for the scraper"""
    parser = argparse.ArgumentParser(description="SuprBay Forum Scraper")

    # Thread-specific arguments
    parser.add_argument("--thread-url", help="URL of a specific thread to scrape")
    parser.add_argument("--thread-list", help="Path to a text file containing thread URLs to scrape")
    
    # Forum section arguments
    parser.add_argument("--forum-section", help="URL of a forum section to scrape all threads from")
    parser.add_argument("--section-pages", type=int, help="Maximum number of section pages to scrape")
    parser.add_argument("--thread-limit", type=int, help="Maximum number of threads to scrape from forum section")
    
    # General arguments
    parser.add_argument("--output-dir", default="data/darkweb", help="Directory to store scraped data")
    parser.add_argument("--tor-port", type=int, default=9150, help="Tor SOCKS proxy port")
    parser.add_argument("--max-pages", type=int, help="Maximum number of pages to scrape per thread")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save HTML for inspection")
    parser.add_argument("--delay-min", type=int, default=8, help="Minimum delay between requests in seconds")
    parser.add_argument("--delay-max", type=int, default=15, help="Maximum delay between requests in seconds")

    args = parser.parse_args()

    if not (args.thread_url or args.thread_list or args.forum_section):
        parser.error("Either --thread-url, --thread-list, or --forum-section must be specified")

    # Adjusting delay based on command line arguments
    delay_range = (args.delay_min, args.delay_max)

    # Initializing the scraper
    scraper = SuprBayScraper(
        output_dir=args.output_dir,
        proxy_port=args.tor_port,
        delay=delay_range,
        debug=args.debug
    )

    # Option 1: Scraping a single thread
    if args.thread_url:
        scraper.scrape_thread(args.thread_url, max_pages=args.max_pages)

    # Option 2: Scraping multiple threads from a list file
    if args.thread_list:
        scrape_from_list(
            args.thread_list,
            output_dir=args.output_dir,
            proxy_port=args.tor_port,
            delay_range=delay_range,
            debug=args.debug,
            max_pages=args.max_pages
        )

    # Option 3: Scraping all threads from a forum section
    if args.forum_section:
        scraper.scrape_forum_section(
            args.forum_section,
            max_section_pages=args.section_pages,
            max_thread_pages=args.max_pages,
            thread_limit=args.thread_limit
        )


if __name__ == "__main__":
    main()