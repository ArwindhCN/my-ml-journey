import re
import numpy as np

def extract_features(url):
    """
    Translates a raw URL string into a mathematical vector: [X1, X2, X3]
    """
    # X1: Length of the URL
    length = len(url)
    
    # X2: Number of hyphens
    hyphens = url.count('-')
    
    # X3: Presence of an IP address (using a basic Regular Expression)
    # This looks for patterns like '192.168.1.1'
    has_ip = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', url) else 0
    
    return [length, hyphens, has_ip]

def process_dataset(url_list):
    """
    Takes a list of URLs and returns a 2D NumPy Matrix.
    """
    matrix = []
    for url in url_list:
        features = extract_features(url)
        matrix.append(features)
        
    return np.array(matrix)

# --- The Extractor in Action ---
if __name__ == "__main__":
    raw_urls = [
        "https://google.com",
        "http://secure-login-update.com",
        "http://192.168.1.1/auth",
        "https://my-bank-verify-account.com"
    ]
    
    print("Raw Data:")
    for url in raw_urls:
        print(f"- {url}")
        
    # Run the engine
    X_matrix = process_dataset(raw_urls)
    
    print("\nExtracted Feature Matrix (X1, X2, X3):")
    print(X_matrix)