import os
import sklearn
from sklearn.datasets import fetch_20newsgroups
from app.recommendation_system import recommendation_system


current_directory = os.getcwd()
desired_directory = f'{current_directory}/.venv/sklearn_data'

newsgroups_train = fetch_20newsgroups(subset='train', data_home=desired_directory)
attributes = dir(newsgroups_train)
#['DESCR', 'data', 'filenames', 'target', 'target_names']


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



        


if __name__ == "__main__":
    main()