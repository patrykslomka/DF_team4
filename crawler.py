import requests
from bs4 import BeautifulSoup
import re
import time
import random
import os
import logging
import datetime
import csv
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bitcoin_crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bitcoin_crawler")

# Cryptocurrency address regex patterns
CRYPTO_PATTERNS = {
    'BTC': re.compile(r'(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}')
}

# Setup session with Tor
session = requests.session()
session.proxies = {
    'http': 'socks5h://127.0.0.1:9150',
    'https': 'socks5h://127.0.0.1:9150'
}
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

# Dummy form data for the websites
DUMMY_FORM_DATA = {
    'email': 'yhfgehfbef@fhefhef.com',
    'name': 'hbhb3f',
    'instructions': 'fuyhebfh3',
    'shipping_address': 'fuyhebfh3',
    'quantity': '1',
}

# Create output directory
os.makedirs("results/html", exist_ok=True)

def save_html(html_content, filename):
    """Save HTML content to a file"""
    with open(f"results/html/{filename}", "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Saved HTML to {filename}")

def extract_wallet_addresses(html_content, url, marketplace):
    """Extract cryptocurrency wallet addresses from HTML content"""
    extracted_data = []
    soup = BeautifulSoup(html_content, 'html.parser')

    # Check all text for wallet addresses
    text = soup.get_text()
    for crypto_type, pattern in CRYPTO_PATTERNS.items():
        for match in pattern.finditer(text):
            wallet_address = match.group(0)
            logger.info(f"Found wallet address: {wallet_address}")

            # Context around the address
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]

            extracted_data.append({
                'marketplace': marketplace,
                'wallet_address': wallet_address,
                'crypto_type': crypto_type,
                'source_url': url,
                'context': context
            })

    # Check JavaScript for wallet addresses
    for script in soup.find_all('script'):
        if script.string:
            for crypto_type, pattern in CRYPTO_PATTERNS.items():
                for match in pattern.finditer(str(script.string)):
                    wallet_address = match.group(0)
                    logger.info(f"Found wallet address in script: {wallet_address}")
                    extracted_data.append({
                        'marketplace': marketplace,
                        'wallet_address': wallet_address,
                        'crypto_type': crypto_type,
                        'source_url': url,
                        'context': "Found in JavaScript"
                    })

    # Check for wallet addresses in QR code images
    for img in soup.find_all('img'):
        for attr in ['alt', 'title', 'data-address', 'data-btc', 'data-content']:
            if img.has_attr(attr):
                attr_value = img[attr]
                for crypto_type, pattern in CRYPTO_PATTERNS.items():
                    for match in pattern.finditer(str(attr_value)):
                        wallet_address = match.group(0)
                        logger.info(f"Found wallet address in QR image {attr}: {wallet_address}")
                        extracted_data.append({
                            'marketplace': marketplace,
                            'wallet_address': wallet_address,
                            'crypto_type': crypto_type,
                            'source_url': url,
                            'context': f"Found in QR image {attr}"
                        })

    return extracted_data

def find_bitcoin_buttons(soup):
    """Find all Bitcoin payment buttons/links on the page"""
    bitcoin_buttons = []

    # Find links with "Pay with Bitcoin" or similar text
    for a_tag in soup.find_all('a'):
        link_text = a_tag.get_text().lower()
        if 'bitcoin' in link_text or 'btc' in link_text:
            if a_tag.has_attr('href'):
                bitcoin_buttons.append({
                    'type': 'link',
                    'url': a_tag['href'],
                    'text': a_tag.get_text().strip()
                })
                logger.info(f"Found Bitcoin link: {a_tag['href']} with text: {a_tag.get_text().strip()}")

    # Find buttons with Bitcoin text
    for button in soup.find_all(['button', 'input']):
        if button.name == 'input' and button.has_attr('value'):
            button_text = button['value'].lower()
        else:
            button_text = button.get_text().lower()

        if 'bitcoin' in button_text or 'btc' in button_text:
            # Check if button is in a form
            form = button.find_parent('form')
            if form:
                form_action = form.get('action', '')
                form_method = form.get('method', 'post').lower()

                # Get all form inputs
                form_inputs = {}
                for input_field in form.find_all(['input', 'textarea']):
                    if input_field.has_attr('name'):
                        form_inputs[input_field['name']] = input_field.get('value', '')

                bitcoin_buttons.append({
                    'type': 'form',
                    'action': form_action,
                    'method': form_method,
                    'inputs': form_inputs,
                    'text': button_text
                })
                logger.info(f"Found Bitcoin form button: {button_text} with action: {form_action}")

    # Find JavaScript buttons
    for element in soup.find_all(['a', 'button', 'div']):
        if element.has_attr('onclick'):
            onclick = element['onclick'].lower()
            if 'bitcoin' in onclick or 'btc' in onclick:
                js_url_match = re.search(r'location\.href\s*=\s*[\'"]([^\'"]+)[\'"]', onclick)
                if js_url_match:
                    bitcoin_buttons.append({
                        'type': 'javascript',
                        'url': js_url_match.group(1),
                        'text': element.get_text().strip()
                    })
                    logger.info(f"Found Bitcoin JavaScript button: {element.get_text().strip()} with URL: {js_url_match.group(1)}")

    # Find elements with Bitcoin-related class names
    for element in soup.find_all(class_=lambda x: x and ('bitcoin' in x.lower() or 'btc' in x.lower())):
        if element.name == 'a' and element.has_attr('href'):
            bitcoin_buttons.append({
                'type': 'link',
                'url': element['href'],
                'text': element.get_text().strip()
            })
            logger.info(f"Found Bitcoin element by class: {element.get_text().strip()} with URL: {element['href']}")

    return bitcoin_buttons

def find_shipping_options(soup):
    """Find shipping option buttons/links on the page"""
    shipping_options = []

    # Look for shipping-related links
    shipping_terms = ['free shipping', 'express shipping', 'dhl', 'courier']

    for a_tag in soup.find_all('a'):
        link_text = a_tag.get_text().lower()
        if any(term in link_text for term in shipping_terms):
            if a_tag.has_attr('href'):
                shipping_options.append({
                    'type': 'link',
                    'url': a_tag['href'],
                    'text': a_tag.get_text().strip()
                })
                logger.info(f"Found shipping link: {a_tag['href']} with text: {a_tag.get_text().strip()}")

    # Look for shipping option forms
    for form in soup.find_all('form'):
        form_text = form.get_text().lower()
        if any(term in form_text for term in shipping_terms):
            form_action = form.get('action', '')
            form_method = form.get('method', 'post').lower()

            # Get all form inputs
            form_inputs = {}
            for input_field in form.find_all(['input', 'textarea']):
                if input_field.has_attr('name'):
                    form_inputs[input_field['name']] = input_field.get('value', '')

            shipping_options.append({
                'type': 'form',
                'action': form_action,
                'method': form_method,
                'inputs': form_inputs
            })
            logger.info(f"Found shipping form with action: {form_action}")

    return shipping_options

def submit_form(form_data, base_url, referrer=None):
    """Submit a form with dummy data"""
    # Fill in the form with dummy data
    form_inputs = form_data.get('inputs', {})

    # Add dummy data to form inputs
    for field_name, field_value in DUMMY_FORM_DATA.items():
        # Find form fields that might match our dummy data keys
        matching_fields = [name for name in form_inputs.keys()
                           if any(key in name.lower() for key in [field_name, field_name.replace('_', '')])]

        # Fill in matching fields
        for match in matching_fields:
            form_inputs[match] = field_value

    # Make sure quantity is 1 if it exists
    for field_name in form_inputs.keys():
        if 'quantity' in field_name.lower():
            form_inputs[field_name] = '1'

    # Get form action URL
    form_action = form_data.get('action', '')
    form_action_url = urljoin(base_url, form_action)

    # Get form method (default to POST)
    form_method = form_data.get('method', 'post').lower()

    # Set referrer if provided
    headers = session.headers.copy()
    if referrer:
        headers['Referer'] = referrer

    # Submit the form
    logger.info(f"Submitting form to {form_action_url} with data: {form_inputs}")

    try:
        if form_method == 'post':
            response = session.post(form_action_url, data=form_inputs, headers=headers)
        else:
            response = session.get(form_action_url, params=form_inputs, headers=headers)

        logger.info(f"Form submission result: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error submitting form: {e}")
        return None

def click_bitcoin_button(button, base_url, referrer=None):
    """Simulate clicking a Bitcoin payment button"""
    button_type = button.get('type', '')

    headers = session.headers.copy()
    if referrer:
        headers['Referer'] = referrer

    try:
        if button_type == 'link':
            url = button.get('url', '')
            button_url = urljoin(base_url, url)
            logger.info(f"Clicking Bitcoin link: {button_url}")
            response = session.get(button_url, headers=headers)
            return response

        elif button_type == 'form':
            return submit_form(button, base_url, referrer)

        elif button_type == 'javascript':
            url = button.get('url', '')
            button_url = urljoin(base_url, url)
            logger.info(f"Clicking Bitcoin JavaScript link: {button_url}")
            response = session.get(button_url, headers=headers)
            return response

        else:
            logger.error(f"Unknown button type: {button_type}")
            return None

    except Exception as e:
        logger.error(f"Error clicking Bitcoin button: {e}")
        return None

def crawl_product_page(url, marketplace_name):
    """Crawl a product page to find Bitcoin wallet addresses"""
    all_extracted_data = []
    visited_urls = set()

    try:
        # 1. Visit the product page
        logger.info(f"Visiting product page: {url}")
        response = session.get(url)
        product_html = response.text
        visited_urls.add(url)

        # Save the product page HTML
        save_html(product_html, f"{marketplace_name}_product_page.html")

        # Extract any wallet addresses from the product page
        product_extracted = extract_wallet_addresses(product_html, url, marketplace_name)
        all_extracted_data.extend(product_extracted)

        # Parse the product page
        product_soup = BeautifulSoup(product_html, 'html.parser')

        # 2. Find shipping options
        shipping_options = find_shipping_options(product_soup)
        logger.info(f"Found {len(shipping_options)} shipping options")

        # Try each shipping option
        for shipping_option in shipping_options:
            time.sleep(random.uniform(2, 5))

            if shipping_option['type'] == 'link':
                shipping_url = urljoin(url, shipping_option['url'])
                logger.info(f"Visiting shipping option: {shipping_url}")
                shipping_response = session.get(shipping_url)
                shipping_html = shipping_response.text
                visited_urls.add(shipping_url)

                save_html(shipping_html, f"{marketplace_name}_shipping_page.html")

                # Extract wallet addresses from shipping page
                shipping_extracted = extract_wallet_addresses(shipping_html, shipping_url, marketplace_name)
                all_extracted_data.extend(shipping_extracted)

                # Look for Bitcoin buttons on the shipping page
                shipping_soup = BeautifulSoup(shipping_html, 'html.parser')
                bitcoin_buttons = find_bitcoin_buttons(shipping_soup)

                # Try each Bitcoin button
                for bitcoin_button in bitcoin_buttons:
                    time.sleep(random.uniform(2, 5))
                    bitcoin_response = click_bitcoin_button(bitcoin_button, shipping_url, shipping_url)

                    if bitcoin_response:
                        bitcoin_html = bitcoin_response.text
                        bitcoin_url = bitcoin_response.url
                        visited_urls.add(bitcoin_url)

                        save_html(bitcoin_html, f"{marketplace_name}_bitcoin_page.html")

                        # Extract wallet addresses from Bitcoin page
                        bitcoin_extracted = extract_wallet_addresses(bitcoin_html, bitcoin_url, marketplace_name)
                        all_extracted_data.extend(bitcoin_extracted)

            elif shipping_option['type'] == 'form':
                shipping_response = submit_form(shipping_option, url, url)

                if shipping_response:
                    shipping_html = shipping_response.text
                    shipping_url = shipping_response.url
                    visited_urls.add(shipping_url)

                    save_html(shipping_html, f"{marketplace_name}_shipping_form_result.html")

                    # Extract wallet addresses from shipping form result
                    shipping_extracted = extract_wallet_addresses(shipping_html, shipping_url, marketplace_name)
                    all_extracted_data.extend(shipping_extracted)

                    # Look for Bitcoin buttons on the shipping form result page
                    shipping_soup = BeautifulSoup(shipping_html, 'html.parser')
                    bitcoin_buttons = find_bitcoin_buttons(shipping_soup)

                    # Try each Bitcoin button
                    for bitcoin_button in bitcoin_buttons:
                        time.sleep(random.uniform(2, 5))
                        bitcoin_response = click_bitcoin_button(bitcoin_button, shipping_url, shipping_url)

                        if bitcoin_response:
                            bitcoin_html = bitcoin_response.text
                            bitcoin_url = bitcoin_response.url
                            visited_urls.add(bitcoin_url)

                            save_html(bitcoin_html, f"{marketplace_name}_bitcoin_page.html")

                            # Extract wallet addresses from Bitcoin page
                            bitcoin_extracted = extract_wallet_addresses(bitcoin_html, bitcoin_url, marketplace_name)
                            all_extracted_data.extend(bitcoin_extracted)

        # 3. Find Bitcoin payment buttons directly on the product page
        bitcoin_buttons = find_bitcoin_buttons(product_soup)
        logger.info(f"Found {len(bitcoin_buttons)} Bitcoin buttons on product page")

        # Try each Bitcoin button
        for bitcoin_button in bitcoin_buttons:
            time.sleep(random.uniform(2, 5))
            bitcoin_response = click_bitcoin_button(bitcoin_button, url, url)

            if bitcoin_response:
                bitcoin_html = bitcoin_response.text
                bitcoin_url = bitcoin_response.url
                visited_urls.add(bitcoin_url)

                save_html(bitcoin_html, f"{marketplace_name}_direct_bitcoin_page.html")

                # Extract wallet addresses from Bitcoin page
                bitcoin_extracted = extract_wallet_addresses(bitcoin_html, bitcoin_url, marketplace_name)
                all_extracted_data.extend(bitcoin_extracted)

                # Fill forms on the Bitcoin page if any
                bitcoin_soup = BeautifulSoup(bitcoin_html, 'html.parser')
                forms = []

                for form in bitcoin_soup.find_all('form'):
                    form_action = form.get('action', '')
                    form_method = form.get('method', 'post').lower()

                    # Get all form inputs
                    form_inputs = {}
                    for input_field in form.find_all(['input', 'textarea']):
                        if input_field.has_attr('name'):
                            form_inputs[input_field['name']] = input_field.get('value', '')

                    forms.append({
                        'type': 'form',
                        'action': form_action,
                        'method': form_method,
                        'inputs': form_inputs
                    })

                # Submit each form
                for form in forms:
                    time.sleep(random.uniform(2, 5))
                    form_response = submit_form(form, bitcoin_url, bitcoin_url)

                    if form_response:
                        form_html = form_response.text
                        form_url = form_response.url
                        visited_urls.add(form_url)

                        save_html(form_html, f"{marketplace_name}_final_page.html")

                        # Extract wallet addresses from form result
                        form_extracted = extract_wallet_addresses(form_html, form_url, marketplace_name)
                        all_extracted_data.extend(form_extracted)

    except Exception as e:
        logger.error(f"Error crawling product page {url}: {e}")

    return all_extracted_data

def crawl_marketplace(url, marketplace_name):
    """Crawl a marketplace to find Bitcoin wallet addresses"""
    all_extracted_data = []

    try:
        # Visit the marketplace homepage
        logger.info(f"Visiting marketplace: {url}")
        response = session.get(url)
        html_content = response.text

        # Save the homepage HTML
        save_html(html_content, f"{marketplace_name}_homepage.html")

        # Extract any wallet addresses from the homepage
        homepage_extracted = extract_wallet_addresses(html_content, url, marketplace_name)
        all_extracted_data.extend(homepage_extracted)

        # Parse the homepage to find product links
        soup = BeautifulSoup(html_content, 'html.parser')
        product_links = []

        # Look for product links in the homepage
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Check if the link might be a product
            if any(term in href.lower() for term in ['product', 'item', 'card', 'visa', 'mastercard', 'paypal']):
                full_url = urljoin(url, href)
                if full_url not in product_links:
                    product_links.append(full_url)

        # If we didn't find any product links, try using common product paths
        if not product_links:
            logger.info("No product links found on homepage. Using predefined paths.")
            predefined_paths = [
                "products/product-prepaid-visa-x1.html",
                "products/product-prepaid-mastercard-x3.html",
                "products/product-paypal-1.html"
            ]
            for path in predefined_paths:
                product_links.append(urljoin(url, path))

        logger.info(f"Found {len(product_links)} product links")

        # Crawl each product page (limit to 3 for testing)
        for product_url in product_links[:3]:
            time.sleep(random.uniform(3, 6))
            product_extracted = crawl_product_page(product_url, marketplace_name)
            all_extracted_data.extend(product_extracted)

    except Exception as e:
        logger.error(f"Error crawling marketplace {marketplace_name}: {e}")

    return all_extracted_data

def save_results_to_csv(results):
    """Save extracted wallet addresses to CSV"""
    if not results:
        logger.info("No results to save")
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/bitcoin_wallets_{timestamp}.csv"

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['marketplace', 'wallet_address', 'crypto_type', 'source_url', 'context']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Saved {len(results)} results to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")
        return None

def check_tor_connection():
    """Check if Tor connection is working"""
    try:
        response = session.get("https://check.torproject.org", timeout=30)
        if "Congratulations. This browser is configured to use Tor" in response.text:
            logger.info("Successfully connected to Tor network")
            return True
        else:
            logger.warning("Not connected to Tor network!")
            return False
    except Exception as e:
        logger.error(f"Error checking Tor connection: {e}")
        return False

def main():
    """Main function to run the crawler"""
    # Check Tor connection
    if not check_tor_connection():
        logger.error("Tor connection failed. Please ensure Tor is running correctly.")
        return

    # Define marketplaces to crawl - test for now
    marketplaces = [
        {
            "name": "Cardzilla",
            "url": "http://cardzilevs4j4nj6uswfwf35oxnp64yrrtazjgap2w3vgoz2pwkp6sqd.onion/"
        },
        {
            "name": "BidenCash",
            "url": "http://bidencgero6anv4hjek7yudvc6ffxfrqihpoqaiwtgf3puj2hvey5lyd.onion/"
        }
    ]

    all_results = []

    # Crawl each marketplace
    for marketplace in marketplaces:
        logger.info(f"Crawling {marketplace['name']}")
        results = crawl_marketplace(marketplace["url"], marketplace["name"])
        all_results.extend(results)
        logger.info(f"Found {len(results)} wallet addresses on {marketplace['name']}")

    # Save all results to CSV
    save_results_to_csv(all_results)
    logger.info("Crawling completed")

if __name__ == "__main__":
    main()