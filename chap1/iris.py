import matplotlib.pyplot as plt
import mglearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    iris_dataset = load_iris()
    print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

    print("Target names: {}".format(iris_dataset["target_names"]))
    print("Feature names: \n{}".format(iris_dataset["feature_names"]))

    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset["data"], iris_dataset["target"], random_state=0
    )

    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    pd.plotting.scatter_matrix(
        iris_dataframe,
        c=y_train,
        figsize=(15, 15),
        marker="o",
        hist_kwds={"bins": 20},
        s=60,
        alpha=0.8,
        cmap=mglearn.cm3,
    )
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y_train)

    score = knn.score(X_test, y_test)
    print("Test set score: {:.2f}".format(score))


if __name__ == "__main__":
    main()
