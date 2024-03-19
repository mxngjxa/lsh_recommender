# lsh_recommender/__init__.py

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define the WordNetLemmatizer
lem = WordNetLemmatizer()

# Download NLTK data and set up stopwords
def download_nltk_data():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

    # Set up stopwords
    global stop_words
    stop_words = set(stopwords.words('english'))

if __name__ == "__main__":
    print("Downloading NLTK data...")
    download_nltk_data()
    print("NLTK data downloaded successfully.")
