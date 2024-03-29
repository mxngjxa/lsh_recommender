import os
import sklearn
from sklearn.datasets import fetch_20newsgroups
from app.recommendation_system import recommendation_system


current_directory = os.getcwd()
desired_directory = f'{current_directory}/.venv/sklearn_data'

newsgroups_train = fetch_20newsgroups(subset='train', data_home=desired_directory)
attributes = dir(newsgroups_train)
#['DESCR', 'data', 'filenames', 'target', 'target_names']
newsgroups_test = fetch_20newsgroups(subset="test", data_home=desired_directory)
print(dir(newsgroups_test))
print(len(newsgroups_test["target"]))



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

        self.lsh_buckets = {}

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




def main():
    shingle_count = 2
    permutations = 2048
    top_k = 5

    print(newsgroups_train[attributes[1]])

    rec_sys = recommendation_system(newsgroups_train[attributes[3]], newsgroups_train[attributes[1]])
    rec_sys.preprocess()
    rec_sys.shingle(shingle_count)
    rec_sys.index(permutations)
    rec_sys.lsh_256()
    query_text = "then she died of a brain tumor, aneurysm, or\nwhatever.  If you can get away without ever ordering"
    print(rec_sys.lsh_buckets)
    sim_articles = rec_sys.query(query_text, top_k)

    print("Top similar articles:", sim_articles)



        


