#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dark Web Cryptocurrency Wallet Crawler

This script implements a secure crawler for academic research purposes,
designed to extract cryptocurrency wallet addresses from Dark Web marketplaces
and forums via the Tor network while maintaining anonymity and adhering to
ethical research guidelines.

For academic research purposes only.
"""

import argparse
import asyncio
import csv
import datetime
import json
import logging
import os
import random
import re
import signal
import sys
import time
from typing import Dict, List, Optional, Set, Tuple, Union

import requests
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from stem import Signal
from stem.control import Controller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("crawler")

# Cryptocurrency address regex patterns
CRYPTO_PATTERNS = {
    'BTC': re.compile(r'(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}'),
    'ETH': re.compile(r'0x[a-fA-F0-9]{40}'),
    'XMR': re.compile(r'4[0-9AB][1-9A-HJ-NP-Za-km-z]{93}')
}

# Default User-Agent rotation list (mimicking different browsers)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
]


class TorController:
    """Manages Tor connection and circuit renewal"""

    def __init__(self, port=9051, password=None):
        """
        Initialize the Tor controller

        Args:
            port: Control port for Tor (default: 9051)
            password: Password for Tor control authentication (if enabled)
        """
        self.port = port
        self.password = password
        self.controller = None

    def connect(self):
        """Establish connection to the Tor controller"""
        try:
            self.controller = Controller.from_port(port=self.port)
            if self.password:
                self.controller.authenticate(password=self.password)
            else:
                self.controller.authenticate()
            logger.info("Successfully connected to Tor controller")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Tor controller: {e}")
            return False

    def renew_tor_circuit(self):
        """Request a new Tor circuit"""
        if not self.controller:
            if not self.connect():
                return False

        try:
            self.controller.signal(Signal.NEWNYM)
            time.sleep(5)  # Wait for the new circuit to be established
            logger.info("Tor circuit renewed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to renew Tor circuit: {e}")
            return False

    def close(self):
        """Close the Tor controller connection"""
        if self.controller:
            self.controller.close()
            logger.info("Tor controller connection closed")


class RequestsWebCrawler:
    """Simple crawler using requests and SOCKS5 proxy for Tor"""

    def __init__(self, tor_proxy="socks5h://127.0.0.1:9050", max_retries=3, timeout=30):
        """
        Initialize the Requests-based crawler

        Args:
            tor_proxy: SOCKS5 proxy URL for Tor
            max_retries: Maximum number of retry attempts per URL
            timeout: Request timeout in seconds
        """
        self.tor_proxy = tor_proxy
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self):
        """Create a requests session with Tor proxy configuration"""
        session = requests.Session()
        session.proxies = {
            'http': self.tor_proxy,
            'https': self.tor_proxy
        }
        return session

    def get_page(self, url: str) -> Optional[str]:
        """
        Fetch a web page through Tor

        Args:
            url: The URL to fetch

        Returns:
            The HTML content of the page or None if failed
        """
        retries = 0
        while retries < self.max_retries:
            try:
                user_agent = random.choice(USER_AGENTS)
                headers = {'User-Agent': user_agent}

                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    logger.info(f"Successfully fetched {url}")
                    return response.text
                else:
                    logger.warning(f"Failed to fetch {url}, status code: {response.status_code}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for {url}: {e}")

            retries += 1
            delay = random.uniform(5, 15)
            logger.info(f"Retrying in {delay:.2f} seconds (attempt {retries}/{self.max_retries})")
            time.sleep(delay)

        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        return None

    def close(self):
        """Close the session"""
        if self.session:
            self.session.close()


class PlaywrightCrawler:
    """Advanced crawler using Playwright for JavaScript-rendered pages and captchas"""

    def __init__(self, tor_proxy="socks5://127.0.0.1:9050", headless=True):
        """
        Initialize the Playwright crawler

        Args:
            tor_proxy: SOCKS5 proxy for Tor
            headless: Whether to run browser in headless mode
        """
        self.tor_proxy = tor_proxy
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.context = None

    async def initialize(self):
        """Initialize Playwright and browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.firefox.launch(
            headless=self.headless,
            proxy={
                "server": self.tor_proxy,
            }
        )
        self.context = await self.browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": 1280, "height": 800}
        )
        logger.info("Playwright browser initialized")

    async def get_page(self, url: str, wait_time: float = 10.0) -> Optional[str]:
        """
        Fetch a web page using Playwright browser

        Args:
            url: The URL to fetch
            wait_time: Time to wait for page to load (in seconds)

        Returns:
            The HTML content of the page or None if failed
        """
        if not self.context:
            await self.initialize()

        try:
            page = await self.context.new_page()
            await page.goto(url, timeout=wait_time * 1000)

            # Wait for the content to load
            await page.wait_for_load_state("networkidle")

            # Additional wait for any potential JavaScript rendering
            await asyncio.sleep(random.uniform(2, 5))

            # Check for common captcha patterns
            captcha_selectors = [
                "input[name='captcha']",
                "img[alt*='captcha' i]",
                "div[class*='captcha' i]",
                "iframe[src*='captcha' i]"
            ]

            for selector in captcha_selectors:
                captcha_element = await page.query_selector(selector)
                if captcha_element:
                    logger.warning(f"Captcha detected on {url}, manual handling required")
                    # In a research environment, you might want to pause for manual captcha handling
                    # This is commented out for automation purposes
                    # await asyncio.sleep(30)  # Wait for manual captcha handling

            content = await page.content()
            await page.close()

            logger.info(f"Successfully fetched {url} with Playwright")
            return content

        except Exception as e:
            logger.error(f"Playwright error for {url}: {e}")
            return None

    async def close(self):
        """Close Playwright browser and context"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Playwright browser closed")


class DarkWebCrawler:
    """Main crawler class that coordinates the crawling process"""

    def __init__(self, config_file: str, output_dir: str = "output"):
        """
        Initialize the Dark Web crawler

        Args:
            config_file: Path to the configuration file
            output_dir: Directory to store output files
        """
        self.config = self._load_config(config_file)
        self.output_dir = output_dir
        self.tor_controller = TorController(
            port=self.config.get("tor_control_port", 9051),
            password=self.config.get("tor_control_password")
        )
        self.requests_crawler = RequestsWebCrawler(
            tor_proxy=self.config.get("tor_proxy", "socks5h://127.0.0.1:9050")
        )
        self.playwright_crawler = None  # Will be initialized asynchronously
        self.circuit_renewal_count = 0
        self.max_requests_per_circuit = self.config.get("max_requests_per_circuit", 10)
        self.extracted_wallets = set()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def _load_config(self, config_file: str) -> Dict:
        """
        Load configuration from a JSON file

        Args:
            config_file: Path to the configuration file

        Returns:
            Dictionary containing configuration
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Default configuration
            return {
                "tor_proxy": "socks5h://127.0.0.1:9050",
                "tor_control_port": 9051,
                "tor_control_password": None,
                "max_requests_per_circuit": 10,
                "request_delay_min": 10,
                "request_delay_max": 30,
                "targets": []
            }

    def _check_circuit_renewal(self):
        """Check if Tor circuit renewal is needed"""
        self.circuit_renewal_count += 1
        if self.circuit_renewal_count >= self.max_requests_per_circuit:
            logger.info(f"Renewing Tor circuit after {self.circuit_renewal_count} requests")
            self.tor_controller.renew_tor_circuit()
            self.circuit_renewal_count = 0

    def extract_crypto_addresses(self, html_content: str, source_url: str, marketplace: str) -> List[Dict]:
        """
        Extract cryptocurrency addresses from HTML content

        Args:
            html_content: HTML content to parse
            source_url: URL of the source page
            marketplace: Name of the marketplace or forum

        Returns:
            List of dictionaries containing extracted wallet information
        """
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements to avoid false positives
        for script in soup(["script", "style"]):
            script.extract()

        # Get text content
        text = soup.get_text(separator=' ', strip=True)

        # Find crypto addresses in the text
        extracted_data = []
        for crypto_type, pattern in CRYPTO_PATTERNS.items():
            for match in pattern.finditer(text):
                wallet_address = match.group(0)

                # Skip if we've already seen this wallet
                if wallet_address in self.extracted_wallets:
                    continue

                self.extracted_wallets.add(wallet_address)

                # Try to find some context (50 chars before and after the wallet)
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text), match.end() + 50)
                context = text[start_pos:end_pos].strip()

                extracted_data.append({
                    'website_name': marketplace,
                    'wallet_address': wallet_address,
                    'cryptocurrency_type': crypto_type,
                    'date_of_extraction': datetime.datetime.now().isoformat(),
                    'source_url': source_url,
                    'additional_context': context
                })

        logger.info(f"Extracted {len(extracted_data)} wallet addresses from {source_url}")
        return extracted_data

    def save_results(self, results: List[Dict], marketplace: str):
        """
        Save extracted wallet data to CSV

        Args:
            results: List of dictionaries containing wallet information
            marketplace: Name of the marketplace or forum
        """
        if not results:
            logger.info(f"No results to save for {marketplace}")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"{marketplace}_{timestamp}.csv")

        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'website_name',
                    'wallet_address',
                    'cryptocurrency_type',
                    'date_of_extraction',
                    'source_url',
                    'additional_context'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

            logger.info(f"Saved {len(results)} results to {filename}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    async def crawl_with_playwright(self, url: str, marketplace: str) -> List[Dict]:
        """
        Crawl a page using Playwright (for JavaScript-rendered content)

        Args:
            url: URL to crawl
            marketplace: Name of the marketplace or forum

        Returns:
            List of dictionaries containing extracted wallet information
        """
        if not self.playwright_crawler:
            self.playwright_crawler = PlaywrightCrawler(
                tor_proxy=self.config.get("tor_proxy", "socks5://127.0.0.1:9050"),
                headless=self.config.get("headless", True)
            )
            await self.playwright_crawler.initialize()

        html_content = await self.playwright_crawler.get_page(url)
        if html_content:
            return self.extract_crypto_addresses(html_content, url, marketplace)
        return []

    def crawl_with_requests(self, url: str, marketplace: str) -> List[Dict]:
        """
        Crawl a page using Requests

        Args:
            url: URL to crawl
            marketplace: Name of the marketplace or forum

        Returns:
            List of dictionaries containing extracted wallet information
        """
        self._check_circuit_renewal()
        html_content = self.requests_crawler.get_page(url)
        if html_content:
            return self.extract_crypto_addresses(html_content, url, marketplace)
        return []

    async def process_target(self, target: Dict) -> List[Dict]:
        """
        Process a target from the configuration

        Args:
            target: Dictionary containing target information

        Returns:
            List of dictionaries containing extracted wallet information
        """
        marketplace = target.get('name', 'unknown')
        base_url = target.get('url')
        urls = target.get('urls', [base_url]) if base_url else target.get('urls', [])
        use_playwright = target.get('use_playwright', False)

        if not urls:
            logger.warning(f"No URLs specified for {marketplace}")
            return []

        results = []
        for url in urls:
            # Add random delay between requests
            delay = random.uniform(
                self.config.get("request_delay_min", 10),
                self.config.get("request_delay_max", 30)
            )
            logger.info(f"Sleeping for {delay:.2f} seconds before next request")
            await asyncio.sleep(delay)

            logger.info(f"Crawling {url} for {marketplace}")
            try:
                if use_playwright:
                    extracted_data = await self.crawl_with_playwright(url, marketplace)
                else:
                    extracted_data = self.crawl_with_requests(url, marketplace)

                results.extend(extracted_data)

            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                continue

        return results

    async def run(self):
        """Run the crawler for all configured targets"""
        # Connect to Tor controller
        self.tor_controller.connect()

        try:
            for target in self.config.get('targets', []):
                results = await self.process_target(target)
                if results:
                    self.save_results(results, target.get('name', 'unknown'))

        finally:
            # Clean up resources
            self.requests_crawler.close()
            if self.playwright_crawler:
                await self.playwright_crawler.close()
            self.tor_controller.close()

        total_extracted = len(self.extracted_wallets)
        logger.info(f"Crawling completed. Total unique wallet addresses extracted: {total_extracted}")
        logger.info(f"Results saved to {self.output_dir}/")


async def main():
    """Main function to run the crawler"""
    parser = argparse.ArgumentParser(description='Dark Web Cryptocurrency Wallet Crawler')
    parser.add_argument('-c', '--config', required=True, help='Path to configuration file')
    parser.add_argument('-o', '--output', default='output', help='Directory for output files')
    args = parser.parse_args()

    try:
        crawler = DarkWebCrawler(args.config, args.output)
        await crawler.run()
    except KeyboardInterrupt:
        logger.info("Crawler stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())