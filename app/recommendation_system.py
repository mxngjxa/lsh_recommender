import re
import os
import nltk
import hashlib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.tokenize import word_tokenize
from datasketch import MinHash, MinHashLSH
try:
    from .optimal_br import OptimalBR
except:
    from optimal_br import OptimalBR


lem = WordNetLemmatizer()

# Set the NLTK_DATA environment variable to the desired directory
current_directory = os.getcwd()
desired_directory = f'{current_directory}/.venv/nltk_data'

nltk.download('stopwords', download_dir=desired_directory)
nltk.download('wordnet', download_dir=desired_directory)
stop_words = set(stopwords.words('english'))

class recommendation_system:
    
    def __init__(self, group, data):
        self.index_raw_data = list()
        i, self.raw_data = group, data
        for group, data in zip(i, self.raw_data):
            #remove the sender information and keep this a simple index, data list
            self.index_raw_data.append([group, data])

    def __repr__(self):
        if not hasattr(self, 'preprocessed') or self.preprocessed is None:
            return f"Raw text of length {len(self.raw_data)}. Awaiting preprocessing."
        elif not hasattr(self, 'shingled') or self.shingled is None:
            return f"Preprocessed text of length {len(self.preprocessed)}. Awaiting shingling."
        elif not hasattr(self, 'signature_matrix') or self.signature_matrix is None:
            return f"Shingled into {self.k}-token shingles. Awaiting MinHash indexing."
        elif not hasattr(self, 'lsh_buckets') or self.lsh_buckets is None:
            return f"MinHash indexed with {self.permutations} permutations. Awaiting Locality Sensitive Hashing (LSH)."
        else:
            return f"Text preprocessed, shingled into {self.k}-token shingles, indexed using MinHash with {self.permutations} permutations, and ready for querying with LSH using {self.b} bands and {self.r} rows per band."


    #clean, remove stopwords, and lemmatize data
    def preprocess(self):
        print("Preprocessing.")
        data = self.index_raw_data
        for i in range(len(data)):
            #remove non-alphanumeric characters
            data[i][1] = data[i][1].split()
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
        print(f"Applying shingling function with {k} tokens.")
        self.k = k
        shingles = list()

        for i in range(len(self.preprocessed)):
            shingles.append([self.preprocessed[i][0], list()])
            for j in range(len(self.preprocessed[i][1]) - self.k):
                #append new shingle as list
                shingle_list = self.preprocessed[i][1][j:j+self.k]
                combined = " ".join([t for t in shingle_list])
                shingles[i][1].append(combined)

        self.shingled_data = shingles
        #print(shingles)
        #[[0, ['This first document', 'first document sure']], [1, ['This document second', 'document second document', 'second document whatever']]]
        print(f"Shingling complete with {self.k} tokens.")


    #use minhashing to permute data into a signature matrix
    def index(self, permutations: int):
        print("MinHashing initiated.")
        self.permutations = permutations
        self.signature_matrix = list()
        
        for i in range(len(self.shingled_data)):
            minhash = MinHash(num_perm=self.permutations)
            for token in self.shingled_data[i][1]:
                minhash.update(token.encode('utf8'))
            hash_values = minhash.digest()
            self.signature_matrix.append([self.shingled_data[i][0], hash_values])
        print("Minhashing processing complete, proceed to LSH.")


    #compute the optimal number of bands and rows per band using a seperate function
    def pre_lsh(self, x: int):
        # Compute optimal b and r based on n
        best_br = OptimalBR()
        self.b, self.r = best_br.br(x)
        return f"{self.b} bands and {self.r} rows per band computed."


    #use lsh_256 to hash items into buckets. LSH processing is complete after this.
    def lsh_256(self, b = None, r = None):
       #complete lsh and returns a dictionary with lsh values as keys and set of documents sorted in as values 
        if self.signature_matrix is None:
            raise ValueError("Signature matrix is not initialized.")
        print("LSH initiated.")

        if b and r:
        # If two parameters are passed, assume they are 'b' and 'r'
            self.b, self.r = b, r
            if self.b * self.r != self.permutations:
                raise ValueError(f"Number of Bands and Rows invalid, product must be equal to {self.permutations}.")
        else:
        #simply automatically calculate the numebr of b and r using the function
            self.pre_lsh(self.permutations)
        self.lsh_buckets = dict()

        for i in range(len(self.signature_matrix)):
            category, current = self.signature_matrix[i]

            for band_index in range(self.b):
                start = band_index * self.r
                band_key = hashlib.sha256(b"".join([line for line in current[start:start + self.r]])).hexdigest()

                if band_key in self.lsh_buckets.keys():
                    self.lsh_buckets[band_key].add(category)
                else:
                    self.lsh_buckets[band_key] = {category}
        print(f"LSH complete with {self.b} bands and {self.r} rows.")
    

    #completes all the previous steps for a unique string and sees which data bucket it would likely fit into.
    def query(self, data_test: str, topk: int):

        if not all(hasattr(self, attr) for attr in ['k', 'permutations', 'b', 'r']):
            raise ValueError("Shingling and LSH parameters are not initialized.")
        
        #initiated document querying
        query_data = data_test.split()
        for i in range(len(query_data)):
            query_data[i] = re.sub(r'\W+', '', query_data[i])
        #stopword removed
        query_data = [w for w in query_data if w not in stop_words]
        #lemmatized
        query_data = [lem.lemmatize(w) for w in query_data]


        #shingling data
        query_shingles = list()

        for i in range(len(query_data) - self.k):
            shingle_list = query_data[i:i+self.k]
            combined = " ".join([t for t in shingle_list])
            query_shingles.append(combined)

        #hash data
        minhash = MinHash(num_perm=self.permutations)
        for token in query_shingles:
            minhash.update(token.encode("utf-8"))
        hash_values = minhash.digest()
        
        #apply lsh and hash into lsh_buckets dictionary
        lsh_keys = list()
        for band_index in range(self.b):
            start = band_index * self.r
            band_key = hashlib.sha256(b"".join([line for line in hash_values[start:start + self.r]])).hexdigest()
            lsh_keys.append(band_key)
        print(lsh_keys)
        
        # Find candidates using LSH
        candidates = self.find_candidates(lsh_keys)[0:topk]
        # Return topK most similar articles
        return candidates


    #after querying, find the most likely candidtates the queried text would fit into.
    def find_candidates(self, query_keys):
        if self.lsh_buckets is None:
            raise ValueError("LSH buckets are not initialized.")
        
        candidates = {}

        # Iterate over each item in the large dataset LSH buckets
        for key, bucket in self.lsh_buckets.items():
            if query_keys == key:
                for item in bucket:
                    if item not in candidates.keys():
                        candidates[item] = 1
                    else:
                        candidates[item] += 1
        
        # Sort the candidates by Jaccard similarity in descending order
        sorted_candidates = sorted(list(candidates.items()), key=lambda x: x[1], reverse=True)
        print("Sorted Candidates", sorted_candidates)

        return sorted_candidates


def main():
    # Sample raw data
    raw_data = [("Sender Information 1", "Email Content 1"), ("Sender Information 2", "Email Content 2, somewier1s_Asjidfj stuuff !!!+£")]
    index = [0, 1]

    # Instantiate recommendation system with sample data
    rec_sys = recommendation_system(index, raw_data)

    # Perform preprocessing
    rec_sys.preprocess()

    # Set shingle size
    k = 2
    rec_sys.shingle(k)

    # Set number of permutations for MinHash
    permutations = 256
    rec_sys.index(permutations)

    # Set parameters for LSH
    #n = 32  # Total number of permutations
    rec_sys.lsh_256()

    # Query text
    query_text = ["This is a test query document."]
    query_index = "index1"

    # Define the value of topK
    topK = 5

    # Query the recommendation system
    top_similar_articles = rec_sys.query(query_index, query_text, topK)

    print("Top similar articles:", top_similar_articles)

    print(rec_sys.lsh_buckets)

if __name__ == "__main__":
    main()