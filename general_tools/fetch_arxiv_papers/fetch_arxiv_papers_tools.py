"""
Fetches and downloads papers from arXiv based on a search query.
This tool is taken from https://programmer.ie/post/deepresearch1/
"""


from smolagents import Tool

import os
import time
import requests
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import re


SEARCH_QUERY= "agent"  # Replace with desired search term or topic
MAX_RESULTS= 50  # Adjust the number of papers you want to download
OUTPUT_FOLDER= "data"  # Folder to store downloaded papers
BASE_URL= "http://export.arxiv.org/api/query?"


class FetchArxivPapersTool(Tool):
    name = "fetch_arxiv_papers"
    description = """
    This is a tool will search arxiv based upn the query. I will return a configurable amount of papers ."""
    inputs = {
        "task": {
            "type": "string",
            "description": "the task category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = "string"


    def sanitize_filename(self, title):
        """Sanitizes a string to be used as a filename."""
        # Remove any characters that are not alphanumeric, spaces, hyphens, or underscores
        return re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")


    def get_filename_from_url(self, url):
        # Parse the URL to get the path component
        parsed_url = urlparse(url)
        # Get the base name from the URL's path
        filename = os.path.basename(parsed_url.path)
        return filename

    def compute_file_hash(self, file_path, algorithm="sha256"):
        """Compute the hash of a file using the specified algorithm."""
        hash_func = hashlib.new(algorithm)

        with open(file_path, "rb") as file:
            # Read the file in chunks of 8192 bytes
            while chunk := file.read(8192):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def fetch_arxiv_papers(self, search_query, max_results=5):
        """Fetches metadata of papers from arXiv using the API."""
        url = f"{BASE_URL}search_query=all:{search_query}&start=0&max_results={max_results}"
        response = requests.get(url)
        response.raise_for_status()
        return response.text


    def parse_paper_links(self, response_text):
        """Parses paper links and titles from arXiv API response XML."""
        import xml.etree.ElementTree as ET

        root = ET.fromstring(response_text)
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            pdf_link = None
            for link in entry.findall("{http://www.w3.org/2005/Atom}link"):
                if link.attrib.get("title") == "pdf":
                    pdf_link = link.attrib["href"] + ".pdf"
                    break
            if pdf_link:
                title = self.get_filename_from_url(pdf_link)
                print(title)
                papers.append((title, pdf_link))
        return papers


    def download_paper(self, title, pdf_link, output_folder):
        """Downloads a single paper PDF."""
        # Create a safe filename
        safe_title = self.sanitize_filename(title)
        filename = os.path.join(output_folder, f"{safe_title}.pdf")
        response = requests.get(pdf_link, stream=True)
        response.raise_for_status()

        # Write the PDF to the specified folder
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"Downloaded: {title}")


    def search(self, search_query, max_results):
        # Create output folder if it doesn't exist
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        # Fetch and parse papers
        print(f"Searching for papers on '{search_query}'...")
        response_text = self.fetch_arxiv_papers(search_query, max_results)
        papers = self.parse_paper_links(response_text)

        # Download each paper
        print(f"Found {len(papers)} papers. Starting download...")
        for title, pdf_link in papers:
            try:
                self.download_paper(title, pdf_link, OUTPUT_FOLDER)
                time.sleep(2)  # Pause to avoid hitting rate limits
            except Exception as e:
                print(f"Failed to download '{title}': {e}")
        print("Download complete!")