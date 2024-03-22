from app.recommendation_system import recommendation_system
import sklearn
from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train')
dir(newsgroups_train)['DESCR', 'data', 'filenames', 'target', 'target_names']


def main():
    print(len(newsgroups_train.data))


if __name__ == "__main__":
    main()