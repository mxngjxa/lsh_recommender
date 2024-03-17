#import libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datasketch import MinHash
import numpy as np
from optimal_br import OptimalBR

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


        def minhash_processing(self, permutations: int):
            num_shingles = len(self.shingled_data)
            self.signature_matrix = np.full((num_shingles, permutations), np.inf)

            for i, shingle in enumerate(shingled_data):
                minhash = MinHash(num_perm=permutations)
                for token in shingle:
                    minhash.update(token.encode('utf8'))
                hash_values = minhash.digest()
                self.signature_matrix[i] = hash_values
        

    def unnamed_function_to_be_defined(self, other, *args):
        if len(args) == 1:
            # If two parameters are passed, assume it is "n"
            n = args
            self.b, self.r = OptimalBR.compute_optimal_br(n)

        elif len(args) == 2:
            # If two parameters are passed, assume they are 'b' and 'r'
            self.b, self.r = args
            
        else:
            # Handle the case where an invalid number of parameters is passed
            raise ValueError("Invalid number of parameters. Expected 1 or 2.")

        

        
    


    
