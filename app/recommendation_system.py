import re
import nltk
import hashlib
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
        self.index_data = list()
        for i, value in enumerate(self.raw_data):
            self.index_data.append([i, value.split()])


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
        data = self.index_data
        for i in range(len(data)):
            for j in range(len(data[i][1])):
                #removed non-alphanumeric characters
                data[i][1][j] = re.sub(r'\W+', '', data[i][1][j])
            #stopword removed
            data[i][1] = [w for w in data[i][1] if w not in stop_words]
            #lemmatized
            data[i][1] = [lem.lemmatize(w) for w in data[i][1]]

        self.preprocessed = data
        #[[0, ['This', 'first', 'document']], [1, ['This', 'document', 'second', 'document']], [2, ['And', 'third', 'one']], [3, ['Is', 'first', 'document']]]
        print("Processing Complete, please apply shingling function.")
    

    #transform document into shingles
    def shingle(self, k: int):
        self.k = k
        shingles = list()

        for i in range(len(self.preprocessed)):
            shingles.append([i, list()])
            for j in range(0, len(self.preprocessed[i][1]) - self.k):
                #append new shingle as list
                shingle_list = self.preprocessed[i][1][j:j+self.k]
                combined = " ".join([t for t in shingle_list])
                shingles[i][1].append(combined)

        self.shingled_data = shingles
        #[[0, ['This first document', 'first document sure']], [1, ['This document second', 'document second document', 'second document whatever']]]
        print(f"Shingling complete with {self.k} tokens.")

    
    def index(self, permutations: int):
        self.permutations = permutations
        self.docs = len(self.shingled_data)
        self.signature_matrix = np.zeros((self.docs, self.permutations))
        
        for i, shingle in enumerate(self.shingled_data):
            minhash = MinHash(num_perm=self.permutations)
            for token in shingle[1]:
                minhash.update(token.encode('utf8'))
            hash_values = minhash.digest()
            self.signature_matrix[i] = hash_values
        print("Minhashing processing complete, proceed to LSH.")


    def pre_lsh(self, x: int):
        # Compute optimal b and r based on n
        best_br = OptimalBR()
        self.b, self.r = best_br.br(x)

    def lsh_256(self, b = None, r = None):
       #complete lsh and returns a dictionary with lsh values as keys and set of documents sorted in as values 
        if self.signature_matrix is None:
            raise ValueError("Signature matrix is not initialized.")

        if b and r:
        # If two parameters are passed, assume they are 'b' and 'r'
            self.b, self.r = b, r
            if self.b * self.r != self.permutations:
                raise ValueError(f"Number of Bands and Rows invalid, product must be equal to {self.permutations}.")
        else:
        #simply automatically calculate the numebr of b and r using the function
            self.pre_lsh(self.permutations)

        print("br", self.b, self.r)
        self.lsh_buckets = {}

        for i in range(self.docs):
            current = self.signature_matrix[i]
            print("current", current)

            for band_index in range(self.b):
                start = band_index * self.r
                band_key = hashlib.sha256(b"".join([line for line in current[start:start + self.r]])).hexdigest()
                print(band_index)

                if band_key in self.lsh_buckets.keys():
                    self.lsh_buckets[band_key].add(i)
                else:
                    self.lsh_buckets[band_key] = {i}
        print(f"LSH complete with {self.b} bands and {self.r} rows.")
    
    def query(self, text_input: str, topk: int, query_key = "alpha"):

        if not all(hasattr(self, attr) for attr in ['k', 'permutations', 'b', 'r']):
            raise ValueError("Shingling and LSH parameters are not initialized.")
        
        item = text_input.split() #already a list
        l = len(item)

        for i in range(l):
            #removed non-alphanumeric characters
            item[i]= re.sub(r'\W+', '', item[i])
        #stopword removed
        item = [w for w in item if w not in stop_words]
        #lemmatized
        item = [lem.lemmatize(w) for w in item]

        shingles = list()
        #shingle data
        for i in range(l - self.k):
            shingles.append(" ".join(item[i:i + self.k]))
        
        #hash data
        minhash = MinHash(num_perm=self.permutations)
        for s in shingles:
            minhash.update(s.encode("utf-8"))
        hashed = minhash.digest()
        
        #apply lsh and hash into lsh_buckets dictionary
        for i in range(self.b):
            start = i * self.r
            item_key = hashlib.sha256(b"".join([line for line in hashed[start:start + self.r]])).hexdigest()

            if item_key in self.lsh_buckets.keys():
                self.lsh_buckets[item_key].add(query_key)
            else:
                self.lsh_buckets[item_key] = {query_key}

        # Find candidates using LSH
        candidates = self.find_candidates(query_key)
        # Return topK most similar articles
        return candidates

    def find_candidates(self, query_key):
        if self.lsh_buckets is None:
            raise ValueError("LSH buckets are not initialized.")

        candidates = {}
        print("self.lsh_buckets", self.lsh_buckets)

        # Iterate over each item in the large dataset LSH buckets
        for key, bucket in self.lsh_buckets.items():
            print("1\n", key, bucket)
            if query_key in bucket:
                for item in bucket:
                    if item not in candidates.keys():
                        candidates[item] = 1
                    else:
                        candidates[item] += 1

        del candidates[query_key]

        # Sort the candidates by Jaccard similarity in descending order
        sorted_candidates = sorted(list(candidates.items()), key=lambda x: x[1], reverse=True)
        print("sorted_candidates", sorted_candidates)

        return sorted_candidates


def main():
    # Sample raw data
    raw_data = [
        "This is the first document are you sure what is going on.",
        "This document is the second document whatever this is fine.",
        "And this is the third one oh boy the document is not long enough.",
        "Is this the first document brush pen is good here?"
    ]

    # Instantiate recommendation system with sample data
    rec_sys = recommendation_system(raw_data)

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
    query_text = "This is a test query document."

    # Define the value of topK
    topK = 5

    # Query the recommendation system
    top_similar_articles = rec_sys.query(query_text, topK)

    print("Top similar articles:", top_similar_articles)

if __name__ == "__main__":
    main()