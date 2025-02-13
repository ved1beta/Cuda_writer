# create_dataset.py

import requests
import json
from tqdm import tqdm
import time
import os

class WikipediaDatasetCreator:
    def __init__(self, output_file="raw_data.jsonl", num_articles=20):
        self.output_file = output_file
        self.num_articles = num_articles
        self.api_url = "https://en.wikipedia.org/w/api.php"
    
    def get_random_articles(self):
        """Fetch random articles from Wikipedia"""
        params = {
            "action": "query",
            "format": "json",
            "list": "random",
            "rnnamespace": 0,  # Only get actual articles
            "rnlimit": self.num_articles
        }
        
        response = requests.get(self.api_url, params=params)
        return response.json()["query"]["random"]
    
    def get_article_content(self, page_id):
        """Fetch content for a specific article"""
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "pageids": page_id,
            "explaintext": True,  # Get plain text instead of HTML
            "exintro": True  # Only get the introduction section
        }
        
        response = requests.get(self.api_url, params=params)
        try:
            return response.json()["query"]["pages"][str(page_id)]["extract"]
        except KeyError:
            return None
    
    def create_dataset(self):
        """Create the dataset file"""
        print(f"Fetching {self.num_articles} random Wikipedia articles...")
        
        # Get random articles
        articles = self.get_random_articles()
        
        # Create the dataset
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for article in tqdm(articles, desc="Downloading articles"):
                content = self.get_article_content(article["id"])
                if content:
                    data = {
                        "text": content,
                        "meta": {
                            "title": article["title"],
                            "source": "wikipedia",
                            "id": article["id"]
                        }
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                
                # Be nice to Wikipedia's API
                time.sleep(1)

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("data", exist_ok=True)
    
    # Create the dataset
    creator = WikipediaDatasetCreator(
        output_file="data/raw_data.jsonl",
        num_articles=20  # Start with 20 articles for testing
    )
    creator.create_dataset()
    
    print("Dataset created successfully!")
    print("Output file: data/raw_data.jsonl")