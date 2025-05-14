import os
import time
import json
import random
import logging
from datetime import datetime
import re
import requests
from bs4 import BeautifulSoup

# Configuring logging in case of errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reddit_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("reddit_scraper")

class RedditScraper:
    """Scraper for Reddit without using API"""

    def __init__(self, output_dir="data/reddit", delay=(3, 8)):
        """Initialize the Reddit scraper"""
        self.output_dir = output_dir
        self.delay = delay
        
        # Creating a session with browser-like headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9'
        })
        
        # Ensuring the output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def random_delay(self):
        """Adding a random delay between requests to avoid rate limiting"""
        delay_time = random.uniform(self.delay[0], self.delay[1])
        logger.info(f"Waiting {delay_time:.2f} seconds before next request...")
        time.sleep(delay_time)

    def clean_text(self, text):
        """Clean text by removing HTML and extra whitespace"""
        if not text:
            return ""
        # Removing HTML tags
        clean = re.sub(r'<.*?>', ' ', text)
        # Removing extra whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def save_data(self, data, filename):
        """Saving scraped data to a JSON file"""
        output_file = os.path.join(self.output_dir, filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Data saved to {output_file}")

    def scrape_subreddit(self, subreddit, limit=25, time_filter='month'):
        """
        Scraping posts from a subreddit
        
        Args:
            subreddit (str): Subreddit name without r/
            limit (int): Maximum number of posts to scrape
            time_filter (str): Time filter (hour, day, week, month, year, all)
            
        Returns:
            list: List of posts with their content
        """
        logger.info(f"Scraping r/{subreddit}...")
        posts = []
        
        # Converting time_filter to Reddit's format if needed
        if time_filter == 'hour':
            time_param = 't=hour'
        elif time_filter == 'day':
            time_param = 't=day'
        elif time_filter == 'week':
            time_param = 't=week'
        elif time_filter == 'month':
            time_param = 't=month'
        elif time_filter == 'year':
            time_param = 't=year'
        else:
            time_param = 't=all'
            
        # Constructing URL
        url = f"https://old.reddit.com/r/{subreddit}/top/?{time_param}" # old.reddit.com is the only way to scrape without using API due to its simple HTML structure
        
        logger.info(f"Starting with URL: {url}")
        
        try:
            # Getting the main page
            response = self.session.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Finding all post elements
            post_elements = soup.select('div.thing.link')
            logger.info(f"Found {len(post_elements)} posts on the first page")
            
            # Processing posts
            posts_processed = 0
            
            for post_element in post_elements:
                if posts_processed >= limit:
                    break
                
                try:
                    # Extracting post ID
                    post_id = post_element.get('data-fullname', '').replace('t3_', '')
                    
                    # Extracting post title
                    title_element = post_element.select_one('a.title')
                    title = title_element.text.strip() if title_element else "No title"
                    
                    # Extracting permalink
                    permalink = title_element.get('href', '') if title_element else ''
                    if permalink.startswith('/r/'):
                        permalink = f"https://old.reddit.com{permalink}"
                    
                    # Extracting author
                    author_element = post_element.select_one('a.author')
                    author = author_element.text.strip() if author_element else "[deleted]"
                    
                    # Creating post data
                    post_data = {
                        "id": post_id,
                        "title": title,
                        "author": author,
                        "permalink": permalink,
                        "content": "",
                        "comments": [],
                        "thread_title": title,  
                        "thread_url": permalink,  
                        "username": author,  
                        "timestamp": datetime.now().strftime("%Y-%m-%d"),  
                        "scrape_time": datetime.now().isoformat()
                    }
                    
                    # Visiting the post page to get content and comments
                    if permalink:
                        logger.info(f"Visiting post: {title[:30]}...")
                        self.random_delay()
                        
                        try:
                            post_response = self.session.get(permalink)
                            post_soup = BeautifulSoup(post_response.text, 'html.parser')
                            
                            # Getting post content
                            selftext_element = post_soup.select_one('div.usertext-body')
                            selftext = ""
                            if selftext_element:
                                paragraphs = selftext_element.select('p')
                                if paragraphs:
                                    selftext = ' '.join([p.text.strip() for p in paragraphs])
                                else:
                                    selftext = selftext_element.text.strip()
                            
                            post_data["content"] = self.clean_text(selftext)
                            
                            # Getting comments
                            comment_elements = post_soup.select('div.comment')
                            comments = []
                            
                            for comment_element in comment_elements[:20]:  # Limiting to 20 comments per post
                                try:
                                    # Extracting author
                                    author_element = comment_element.select_one('a.author')
                                    author = author_element.text.strip() if author_element else "[deleted]"
                                    
                                    # Extracting content
                                    content_element = comment_element.select_one('div.usertext-body')
                                    content = ""
                                    if content_element:
                                        paragraphs = content_element.select('p')
                                        if paragraphs:
                                            content = ' '.join([p.text.strip() for p in paragraphs])
                                        else:
                                            content = content_element.text.strip()
                                    
                                    comments.append({
                                        "author": author,
                                        "content": self.clean_text(content)
                                    })
                                    
                                except Exception as e:
                                    logger.error(f"Error processing comment: {e}")
                                    continue
                            
                            post_data["comments"] = comments
                            
                        except Exception as e:
                            logger.error(f"Error visiting post page: {e}")
                    
                    posts.append(post_data)
                    posts_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing post: {e}")
                    continue
                
                self.random_delay()
            
            # Saving the data
            if posts:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{subreddit}_posts_{timestamp}.json"
                self.save_data(posts, filename)
                logger.info(f"Saved {len(posts)} posts from r/{subreddit}")
            
            return posts
            
        except Exception as e:
            logger.error(f"Error scraping subreddit r/{subreddit}: {e}")
            return []

def scrape_reddit(config, skip_reddit=False):
    """
    Scraping data from Reddit subreddits

    Args:
        config (dict): Configuration dictionary
        skip_reddit (bool): Whether to skip Reddit scraping

    Returns:
        bool: True if successful, False otherwise
    """
    if skip_reddit:
        logger.info("Skipping Reddit scraping as requested")
        return True

    try:
        logger.info("Starting Reddit scraping...")
        
        # Getting Reddit configuration
        if isinstance(config, str):
            # If config is a file path, loading it
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

        subreddits = reddit_config.get('subreddits', [])
        scraping_params = reddit_config.get('scraping_params', {})
        limit = scraping_params.get('limit', 25)
        time_filter = scraping_params.get('time_filter', 'month')
        
        # Creating output directory if it doesn't exist
        output_dir = os.path.join('data', 'reddit')
        os.makedirs(output_dir, exist_ok=True)

        # Initializing the Reddit scraper
        reddit_scraper = RedditScraper(output_dir=output_dir)

        # Scraping each subreddit
        for subreddit in subreddits:
            try:
                logger.info(f"Scraping subreddit: r/{subreddit}")
                reddit_scraper.scrape_subreddit(
                    subreddit=subreddit, 
                    limit=limit, 
                    time_filter=time_filter
                )
                # Adding a delay between subreddits
                time.sleep(random.uniform(5, 10))
            except Exception as e:
                logger.error(f"Error scraping subreddit r/{subreddit}: {e}")
                continue

        logger.info("Reddit scraping completed")
        return True

    except Exception as e:
        logger.error(f"Error in Reddit scraping pipeline: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 2:
        print("Usage: python reddit_scraper.py config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    scrape_reddit(config_path)