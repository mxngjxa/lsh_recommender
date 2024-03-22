import os
import sklearn
from sklearn.datasets import fetch_20newsgroups
from app.recommendation_system import recommendation_system


current_directory = os.getcwd()
desired_directory = f'{current_directory}/.venv/sklearn_data'

newsgroups_train = fetch_20newsgroups(subset='train', data_home=desired_directory)
attributes = dir(newsgroups_train)


def main():
    targets = ['DESCR', 'data', 'filenames', 'target', 'target_names']
    print(newsgroups_train)


if __name__ == "__main__":
    main()