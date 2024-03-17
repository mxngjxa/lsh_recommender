#import libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class recommendation_system:
    
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.preprocessed = None
        self.k = None
        self.shingled = None

    def __repr__(self):
        if not self.preprocessed:
            return f"Raw text of length {len(self.raw_data)}. Awaiting preprocessing."
        elif not self.shingled:
            return f"Raw text of length {len(self.raw_data)}, processed into a list of length {len(self.processed)}. Awaiting further action."
        else:
            return f"Processed text of length {len(self.processed)} shingled into {self.k} shingles. Awaiting further action"

    def preprocess(self):
        #tokenize words
        tokenized = word_tokenize(self.raw_data)
        tokenized = tokenized.lower()
        print("Tokenization Complete")

        #remove non-ascii characters
        ascii_filtered = ''.join([x for x in tokenized if x.isascii()])

        #remove stopwords
        stopwords_removed = [w for w in ascii_filtered if w not in stop_words]
        print("Stopwords Removed")

        #lemmatization
        lemmatized = [lem.lemmatize(w) for w in stopwords_removed]
        print("Lemmatization Complete.")

        self.processed = lemmatized
        print("Processing Complete, please apply shingling function.")
    
    def shingle(self, k:int):
        self.k = k
        shingles = list()
        for i in range(0, len(self.preprocessed) - self.k):
            shingles.append(self.preprocessed[i:i+self.k])
        self.shingled = tuple(shingles)
        print(f"Shingling complete with {self.k} tokens/shingle.")

    def process(*args):
        if len(args) == 1:
            # If two parameters are passed, assume it is "n"
            n = args

        elif len(args) == 2:
            # If two parameters are passed, assume they are 'b' and 'r'
            b, r = args
        else:
            # Handle the case where an invalid number of parameters is passed
            raise ValueError("Invalid number of parameters. Expected 1 or 2.")
    


    
