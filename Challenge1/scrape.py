import os
import time
import random
import requests
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Settings
search_term = "red paint splatter"
num_images = 100
output_dir = "paint_splatter_images"
os.makedirs(output_dir, exist_ok=True)

# Chrome options - simpler configuration
options = Options()
options.add_argument("--headless=new")  # Updated headless mode for newer Chrome
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1920,1080")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

print("Starting browser...")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Function to scrape images from a search engine
def scrape_images_from_source(source):
    image_urls = []
    
    try:
        if source == "google":
            # Google Images
            print(f"Searching for '{search_term}' on Google Images...")
            encoded_query = urllib.parse.quote(search_term)
            driver.get(f"https://www.google.com/search?q={encoded_query}&tbm=isch")
            time.sleep(3)  # Wait for page to load
            
            # Scroll down to load more images
            print("Scrolling to load more images...")
            for _ in range(10):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)
                
                # Try to click "Show more results" if it exists
                try:
                    show_more = driver.find_element(By.CSS_SELECTOR, ".mye4qd")
                    if show_more.is_displayed():
                        show_more.click()
                        time.sleep(2)
                except:
                    pass
            
            # Extract image URLs - look for thumbnails
            print("Finding image elements...")
            img_elements = driver.find_elements(By.CSS_SELECTOR, "img.rg_i, img.Q4LuWd")
            print(f"Found {len(img_elements)} image elements on Google")
            
            # Extract the source URLs
            for img in img_elements:
                try:
                    # Try to get image URL from src or data-src attribute
                    src = img.get_attribute("src") or img.get_attribute("data-src")
                    
                    # Sometimes the URL is in the dataset
                    if not src or not src.startswith("http"):
                        # Try extracting from dataset if available
                        src = img.get_attribute("data-iurl")
                    
                    if src and src.startswith("http"):
                        image_urls.append(src)
                        if len(image_urls) >= num_images:
                            break
                except Exception as e:
                    print(f"Error extracting image source: {e}")
                    
        elif source == "bing":
            # Bing Images
            print(f"Searching for '{search_term}' on Bing Images...")
            encoded_query = urllib.parse.quote(search_term)
            driver.get(f"https://www.bing.com/images/search?q={encoded_query}")
            time.sleep(3)  # Wait for page to load
            
            # Scroll down to load more images
            print("Scrolling to load more images...")
            for _ in range(5):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            # Extract image URLs
            print("Finding image elements...")
            img_elements = driver.find_elements(By.CSS_SELECTOR, ".mimg, .iusc img")
            print(f"Found {len(img_elements)} image elements on Bing")
            
            for img in img_elements:
                try:
                    src = img.get_attribute("src")
                    
                    # Try to get the full-size image URL from parent element's m attribute
                    if not src or "base64" in src:
                        parent = img.find_element(By.XPATH, "./ancestor::a[contains(@class, 'iusc')]")
                        m_attr = parent.get_attribute("m")
                        if m_attr:
                            import json
                            m_data = json.loads(m_attr)
                            if "murl" in m_data:
                                src = m_data["murl"]
                    
                    if src and src.startswith("http") and "base64" not in src:
                        image_urls.append(src)
                        if len(image_urls) >= num_images:
                            break
                except Exception as e:
                    print(f"Error extracting Bing image source: {e}")
                    
        elif source == "duckduckgo":
            # DuckDuckGo Images
            print(f"Searching for '{search_term}' on DuckDuckGo Images...")
            encoded_query = urllib.parse.quote(search_term)
            driver.get(f"https://duckduckgo.com/?q={encoded_query}&t=h_&iax=images&ia=images")
            time.sleep(5)  # DuckDuckGo needs more time to load
            
            # Click on the images tab if not already on it
            try:
                img_tab = driver.find_element(By.CSS_SELECTOR, ".js-zci-link--images")
                img_tab.click()
                time.sleep(2)
            except:
                pass  # Already on images tab
                
            # Scroll to load more images
            print("Scrolling to load more images...")
            for _ in range(8):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)
            
            # Extract image URLs
            print("Finding image elements...")
            img_elements = driver.find_elements(By.CSS_SELECTOR, ".tile--img__img")
            print(f"Found {len(img_elements)} image elements on DuckDuckGo")
            
            for img in img_elements:
                try:
                    src = img.get_attribute("src")
                    data_src = img.get_attribute("data-src")
                    
                    # Sometimes the URL is in data-src
                    if data_src and data_src.startswith("http"):
                        image_urls.append(data_src)
                    elif src and src.startswith("http") and "base64" not in src:
                        image_urls.append(src)
                        
                    if len(image_urls) >= num_images:
                        break
                except Exception as e:
                    print(f"Error extracting DuckDuckGo image source: {e}")
                    
    except Exception as e:
        print(f"Error accessing {source}: {e}")
        
    print(f"Found {len(image_urls)} image URLs from {source}")
    return image_urls

# Try all sources until we get enough images
all_image_urls = []
sources = ["bing", "google", "duckduckgo"]  # Bing first as it's usually more reliable

for source in sources:
    if len(all_image_urls) >= num_images:
        break
        
    source_urls = scrape_images_from_source(source)
    all_image_urls.extend(source_urls)
    print(f"Total image URLs collected so far: {len(all_image_urls)}")

# Clean up browser
driver.quit()

# Deduplicate URLs
all_image_urls = list(dict.fromkeys(all_image_urls))
print(f"Found {len(all_image_urls)} unique image URLs")

# Download images
print("Downloading images...")
successful_downloads = 0

for i, url in enumerate(all_image_urls):
    if successful_downloads >= num_images:
        break
        
    try:
        print(f"Downloading image {i+1}/{min(len(all_image_urls), num_images)} from {url[:50]}...")
        
        # Use requests with proper headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Check if it's actually an image
            content_type = response.headers.get('Content-Type', '')
            if 'image' in content_type:
                # Determine file extension based on content type
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = '.jpg'
                elif 'png' in content_type:
                    ext = '.png'
                elif 'gif' in content_type:
                    ext = '.gif'
                elif 'webp' in content_type:
                    ext = '.webp'
                else:
                    ext = '.jpg'  # Default to jpg
                
                file_path = os.path.join(output_dir, f"{successful_downloads+1:03d}{ext}")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                successful_downloads += 1
                print(f"Successfully downloaded image {successful_downloads}/{num_images}")
                
                # Rate limiting
                time.sleep(random.uniform(0.5, 1.5))
            else:
                print(f"Skipping non-image content: {content_type}")
        else:
            print(f"Failed to download: HTTP status {response.status_code}")
    except Exception as e:
        print(f"Error downloading image: {e}")

print(f"Script completed. Downloaded {successful_downloads} images to {output_dir}/")