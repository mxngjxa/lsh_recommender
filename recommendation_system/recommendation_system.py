#import libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class recommendation_system:
    
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.processed = None

    def preprocess(self):
        #tokenize words
        tokenized = word_tokenize(self.raw_data)
        tokenized = tokenized.lower()
        print("Tokenization Complete")

        #remove stopwords
        stopwords_removed = [w for w in word_tokens if w not in stop_words]
        print("Stopwords Removed")

        #lemmatization
        lemmatized = [lem.lemmatize(w) for w in stopwords_removed]
        print("Lemmatization Complete.")

        self.processed = lemmatized
        print("Processing Complete")

    def __repr__(self):
        if not self.processed:
            return f"Raw text of length {len(self.raw_data)}. Awaiting processing."
        else:
            return f"Raw text of length {len(self.raw_data)}, processed into a list of length {len(self.processed)}. Awaiting further action."



    
