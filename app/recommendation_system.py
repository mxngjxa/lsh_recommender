import hashlib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.tokenize import word_tokenize
from datasketch import MinHash, MinHashLSH
from optimal_br import OptimalBR


lem = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

class recommendation_system:
    
    def __init__(self, raw_data):
        self.raw_data = raw_data
        self.index_data = []
        for i, value in enumerate(self.raw_data):
            self.index_data.append([i, value.split()])
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
        #remove non-alphabetical characters
        data = self.index_data
        for i in range(len(data)):
            for j in range(len(data[i][1])):
                data[i][1][j] = re.sub(r'\W+', '', data[i][1][j])
            #stopword removed
            data[i][1] = [w for w in data[i][1] if w not in stop_words]
            #lemmatized
            data[i][1] = [lem.lemmatize(w) for w in data[i][1]]
            
        self.preprocessed = data
        print("Processing Complete, please apply shingling function.")
    

    #transform document into shingles
    def shingle(self, k: int):
        self.k = k
        shingles = list()
        for i in range(0, len(self.preprocessed) - self.k):
            shingles.append(self.preprocessed[i:i+self.k])
        self.shingled_data = tuple(shingles)
        print(f"Shingling complete with {self.k} tokens.")

    
    def minhash_processing(self, permutations: int):
        self.permutations = permutations
        self.signature_matrix = np.full((len(self.shingled_data), self.permutations), np.inf)

        print(self.shingled_data)
        for i, shingle in enumerate(self.shingled_data):
            minhash = MinHash(num_perm=self.permutations)
            for token in shingle:
                minhash.update(token.encode('utf8'))
            hash_values = minhash.digest()
            self.signature_matrix[i] = hash_values
        
        print("Minhashing processing complete, proceed to LSH.")
        # print signature matrix
        print("signature_matrix", self.signature_matrix)


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
        
        self.b, self.r = int(self.b), int(self.r)
        # Apply LSH to the current dataset
        self.compute_lsh_buckets()
        print(self.lsh_buckets)

    def compute_lsh_buckets(self):
        if self.signature_matrix is None:
            raise ValueError("Signature matrix is not initialized.")

        num_shingles, num_permutations = self.signature_matrix.shape
        self.lsh_buckets = {}

        for i in range(num_shingles):
            buckets = {}
            signature = self.signature_matrix[i]

            for band_index in range(self.b):
                band_hash_values = [hashlib.sha256(bytes(signature[column_index:column_index + self.r])).digest() for column_index in range(num_permutations - self.r)]
                band_key = hashlib.sha256(b"".join(band_hash_values)).hexdigest()

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
    
    def query(self, text_input: str, topk: int):
        if not all(hasattr(self, attr) for attr in ['k', 'permutations', 'b', 'r']):
            raise ValueError("Shingling and LSH parameters are not initialized.")

        tokenized = word_tokenize(text_input.lower())
        ascii_filtered = ''.join([x for x in tokenized if x.isascii()])
        stopwords_removed = [w for w in ascii_filtered if w not in stop_words]
        lemmatized = [lem.lemmatize(w) for w in stopwords_removed]
        shingles = []
        for i in range(0, len(lemmatized) - self.k):
            shingles.append(lemmatized[i:i + self.k])
        shingles = tuple(shingles)
        print(f"Preprocessing/shingling complete with {self.k} tokens.")

        text_input_signature = np.full((len(shingles), self.permutations), np.inf)

        for i, shingle in enumerate(shingles):
            minhash = MinHash(num_perm=self.permutations)
            for token in shingle:
                minhash.update(token.encode('utf8'))
            hash_values = minhash.digest()
            text_input_signature[i] = hash_values
        print("Minhashing processing complete, proceed to LSH.")

        num_shingles, num_permutations = text_input_signature.shape
        query_lsh_buckets = {}

        for i in range(num_shingles):
            buckets = {}
            signature = text_input_signature[i]

            for band_index in range(self.b):
                band_hash_values = [hashlib.sha256(bytes(signature[column_index:column_index + self.r])).digest() for column_index in range(num_permutations - self.r)]
                band_key = hashlib.sha256(b"".join(band_hash_values)).hexdigest()

                if band_key in buckets:
                    buckets[band_key].add(i)
                else:
                    buckets[band_key] = {i}

            query_lsh_buckets[i] = buckets
        
        print("query_lsh_buckets", query_lsh_buckets)

        # Find candidates using LSH
        candidates = self.find_candidates(query_lsh_buckets)
        # Return topK most similar articles
        return self.top_k_similar_articles(candidates, topk)

    def find_candidates(self, query_lsh_buckets):
        if query_lsh_buckets is None or self.lsh_buckets is None:
            raise ValueError("LSH buckets are not initialized.")

        candidates = {}

        # Iterate over each item in the large dataset LSH buckets
        for key, buckets in self.lsh_buckets.items():
            print("1\n", key, buckets)
            # Iterate over each bucket in the large dataset LSH buckets
            for band_key, bucket in buckets.items():
                print("2\n", band_key, bucket)
                # Compute Jaccard similarity between the query MinHash and the MinHash of the current item
                jaccard_similarity = band_key.jaccard(item_minhash)

                # Add the item ID and its Jaccard similarity to the candidates dictionary
                if key in candidates:
                    candidates[key] += jaccard_similarity
                else:
                    candidates[key] = jaccard_similarity

        # Sort the candidates by Jaccard similarity in descending order
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        return sorted_candidates


# Sample raw data
raw_data = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Instantiate recommendation system with sample data
rec_sys = recommendation_system(raw_data)

# Perform preprocessing
rec_sys.preprocess()

# Set shingle size
k = 3
rec_sys.shingle(k)

# Set number of permutations for MinHash
permutations = 128
rec_sys.minhash_processing(permutations)

# Set parameters for LSH
n = 128  # Total number of permutations
rec_sys.lsh(n)

# Query text
query_text = "This is a test query document."

# Define the value of topK
topK = 5

# Query the recommendation system
top_similar_articles = rec_sys.query(query_text, topK)

print("Top similar articles:", top_similar_articles)