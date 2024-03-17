#import libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datasketch import MinHash, MinHashLSH
import numpy as np
from optimal_br import OptimalBR

lem = WordNetLemmatizer()

#stopwords set as english, change language as needed
LANGUAGE = "english"
with open(f"stopwords/{LANGUAGE}", "r") as file:
    stop_words = set(file.read().split())

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
            return f"Processed text of length {len(self.preprocessed)}. Awaiting shingling."
        elif not self.signature_matrix:
            return f"Shingled into {self.k} shingles. Awaiting processing."
        else:
            return f"Processed and indexed. Ready for recommendation."


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
    

    #transform document into shingles
    def shingle(self, k: int):
        self.k = k
        shingles = list()
        for i in range(0, len(self.preprocessed) - self.k):
            shingles.append(self.preprocessed[i:i+self.k])
        self.shingled_data = tuple(shingles)
        print(f"Shingling complete with {self.k} tokens/shingle.")

    
    def minhash_processing(self, permutations: int):
        self.permutations = permutations
        self.signature_matrix = np.full((len(self.shingled_data), self.permutations), np.inf)

        for i, shingle in enumerate(self.shingled_data):
            minhash = MinHash(num_perm=self.permutations)
            for token in shingle:
                minhash.update(token.encode('utf8'))
            hash_values = minhash.digest()
            self.signature_matrix[i] = hash_values
        
        print("Minhashing processing complete, proceed to LSH.")


    def lsh(self, *args):
        if len(args) == 1:
            # If one parameter is passed, assume it is "n" (total number of permutations)
            n = args[0]
            # Compute optimal b and r based on n
            self.b, self.r = OptimalBR.compute_optimal_br(n)
        elif len(args) == 2:
            # If two parameters are passed, assume they are 'b' and 'r'
            self.b, self.r = args
        else:
            # Handle the case where an invalid number of parameters is passed
            raise ValueError("Invalid number of parameters. Expected 1 or 2.")

        # Apply LSH to the current dataset
        self.compute_lsh_buckets()

    def compute_lsh_buckets(self):
        if self.signature_matrix is None:
            raise ValueError("Signature matrix is not initialized.")

        num_shingles, num_permutations = self.signature_matrix.shape
        self.lsh_buckets = {}

        for i in range(num_shingles):
            buckets = {}
            signature = self.signature_matrix[i]

            for band_index in range(self.b):
                band_hash_values = [hash(signature[column_index:column_index + self.r]) for column_index in range(num_permutations - self.r + 1)]
                band_key = hashlib.sha256(bytes(band_hash_values)).hexdigest()

                if band_key in buckets:
                    buckets[band_key].add(i)
                else:
                    buckets[band_key] = {i}

            self.lsh_buckets[i] = buckets

    def index(self, article_id, signature):
        if self.signature_matrix is None:
            raise ValueError("Signature matrix is not initialized.")

        if article_id >= len(self.signature_matrix):
            raise ValueError("Article ID exceeds signature matrix size.")

        self.signature_matrix[article_id] = signature
