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

        #one hot encoding
        one_hot_encoded_list = np.full((self.shingle_count), 0)

        for i in range(len(self.shingle_array)):
            if self.shingle_array[i] in query_shingles:
                one_hot_encoded_list[i] = 1
        
        #create a mini permutation matrix
        signature_list = np.full((self.permutations), 0)

        for i in range(self.permutations):
            for j in range(self.shingle_count):
                s = np.where(np.isclose(self.permutation_matrix[i], j))[0][0]
                if one_hot_encoded_list[s] == 1:
                    signature_list[i] = self.permutation_matrix[i, s]
        
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