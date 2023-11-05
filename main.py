# This is a sample Python script.
import time
from math import sqrt

import numpy as np
import scipy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import MDS, locally_linear_embedding, LocallyLinearEmbedding
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import f1_score, euclidean_distances, homogeneity_score
from sklearn.random_projection import GaussianRandomProjection
from yellowbrick.features import manifold_embedding
from yellowbrick.model_selection import LearningCurve, ValidationCurve
from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt


def main():

    print('clustering1')
    clustering1()
    print('clustering2')
    clustering2()
    print('pca1')
    pca1()
    print('pca2')
    pca2()
    print('ica1')
    ica1()
    print('ica2')
    ica2()
    print('rp1')
    rp1()
    print('rp2')
    rp2()
    print('manifold1')
    manifold1()
    print('manifold2')
    manifold2()

def manifold1():
    data = pd.read_csv('./adult/adult.data')
    X = data.iloc[:3000, :14]
    X = OneHotEncoder(handle_unknown='ignore').fit_transform(X)
    y = data.iloc[:3000, -1]
    y = LabelEncoder().fit_transform(y)

    # code modified from https://www.scikit-yb.org/en/latest/api/features/manifold.html
    manifold_embedding(X.toarray(), y, manifold="modified", random_state=0, n_neighbors=10, kwargs={'n_jobs': -1}).show()
    manifold_embedding(X.toarray(), y, manifold="modified", projection=3, random_state=0, n_neighbors=10, kwargs={'n_jobs': -1}).show()
    # end

    g = locally_linear_embedding(X.toarray(), n_components=3, random_state=0, n_jobs=-1, method='modified', n_neighbors=10)
    print(g)

    a = LocallyLinearEmbedding(n_components=3, random_state=0, n_jobs=-1, method='modified', n_neighbors=10).fit_transform(X.toarray())

    x = range(1,21, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('cluster distance')
    for i in x:
        g = GaussianMixture(n_components=i, random_state=0,init_params='random_from_data').fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.means_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist =  tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    x = range(1, 21, 3)
    z = []
    plt.xlabel('clusters')
    plt.ylabel('cluster distance')
    for i in x:
        g = KMeans(n_clusters=i, random_state=0).fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.cluster_centers_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist = tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    g = GaussianMixture(n_components=7, random_state=0, init_params='random_from_data').fit(a)
    print('EM')
    print(homogeneity_score(y.squeeze(), g.predict(a)))

    g = KMeans(n_clusters=8, random_state=0).fit(a)
    print('KMeans')
    print(homogeneity_score(y.squeeze(), g.predict(a)))

def manifold2():
    # code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    # end
    y = LabelEncoder().fit_transform(y)
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.9, random_state=0)
    # end

    # code modified from https://www.scikit-yb.org/en/latest/api/features/manifold.html
    manifold_embedding(X_train, y_train, manifold="modified", random_state=0, n_neighbors=10, kwargs={'n_jobs': -1}).show()
    manifold_embedding(X_train, y_train, manifold="modified", projection=3, random_state=0, n_neighbors=10, kwargs={'n_jobs': -1}).show()
    # end

    g = locally_linear_embedding(X_train, n_components=3, random_state=0, n_jobs=-1, method='modified', n_neighbors=10)
    print(g)

    a = LocallyLinearEmbedding(n_components=3, random_state=0, n_jobs=-1, method='modified',
                               n_neighbors=10).fit_transform(X_train)

    x = range(1,21, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('cluster distance')
    for i in x:
        g = GaussianMixture(n_components=i, random_state=0,init_params='random_from_data').fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.means_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist =  tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    x = range(1, 21, 3)
    z = []
    plt.xlabel('clusters')
    plt.ylabel('cluster distance')
    for i in x:
        g = KMeans(n_clusters=i, random_state=0).fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.cluster_centers_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist = tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    g = GaussianMixture(n_components=7, random_state=0, init_params='random_from_data').fit(a)
    print('EM')
    print(homogeneity_score(y_train, g.predict(a)))

    g = KMeans(n_clusters=11, random_state=0).fit(a)
    print('KMeans')
    print(homogeneity_score(y_train, g.predict(a)))

    print('NNReduction')

    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                        cv=StratifiedKFold(n_splits=5), n_jobs=10, scoring='f1_micro')
    vis.fit(a, np.ravel(y_train))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                          param_name="max_iter",
                          param_range=np.arange(0, 210, 10), cv=StratifiedKFold(n_splits=5), scoring='f1_micro',
                          n_jobs=10)

    vis.fit(a, np.ravel(y_train))
    vis.show()
    # end

    v = MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95))

    strt = time.time()
    v.fit(a, y_train)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(a)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, average='micro'))
    print(' ')

def rp1():
    data = pd.read_csv('./adult/adult.data')
    X = data.iloc[:10000, :14]
    X = OneHotEncoder(handle_unknown='ignore').fit_transform(X)
    y = data.iloc[:10000, -1]

    x = range(1,31, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('reconstruction')
    for i in x:
        print(i)
        a = GaussianRandomProjection(n_components=i, random_state=0).fit(X.toarray())
        # code from https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn, metric from https://core.ac.uk/download/pdf/148661309.pdf
        g = a.transform(X.toarray())
        x_rec = a.inverse_transform(g)
        an = (X - x_rec)
        an = np.multiply(an,an)
        reconstruction = sqrt(an.sum())
        print(reconstruction)
        # End
        z.append(reconstruction)

    plt.plot(x, z)
    plt.show()

    a = GaussianRandomProjection(n_components=10, random_state=0).fit(X.toarray())
    # code from https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn, metric from https://core.ac.uk/download/pdf/148661309.pdf
    g = a.transform(X.toarray())
    x_rec = a.inverse_transform(g)
    an = (X - x_rec)
    an = np.multiply(an,an)
    print(sqrt(an.sum()))
    # End

    a = GaussianRandomProjection(n_components=28, random_state=0).fit_transform(X.toarray())

    x = range(1,21, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('cluster distance')
    for i in x:
        g = GaussianMixture(n_components=i, random_state=0,init_params='random_from_data').fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.means_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist =  tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    x = range(1, 21, 3)
    z = []
    plt.xlabel('clusters')
    plt.ylabel('cluster distance')
    for i in x:
        g = KMeans(n_clusters=i, random_state=0).fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.cluster_centers_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist = tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    g = GaussianMixture(n_components=10, random_state=0, init_params='random_from_data').fit(a)
    print('EM')
    print(homogeneity_score(y.squeeze(), g.predict(a)))

    g = KMeans(n_clusters=11, random_state=0).fit(a)
    print('KMeans')
    print(homogeneity_score(y.squeeze(), g.predict(a)))

def rp2():
    # code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    # end
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)
    # end

    x = range(1, 31, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('reconstruction')
    for i in x:
        print(i)
        a = GaussianRandomProjection(n_components=i, random_state=0).fit(X_train)
        # code from https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn, metric from https://core.ac.uk/download/pdf/148661309.pdf
        g = a.transform(X_train)
        x_rec = a.inverse_transform(g)
        an = (X_train - x_rec)
        an = np.multiply(an, an)
        reconstruction = sqrt(an.sum())
        print(reconstruction)
        # End
        z.append(reconstruction)

    plt.plot(x, z)
    plt.show()

    a = GaussianRandomProjection(n_components=12, random_state=0).fit(X_train)
    # code from https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn, metric from https://core.ac.uk/download/pdf/148661309.pdf
    g = a.transform(X_train)
    x_rec = a.inverse_transform(g)
    an = (X_train - x_rec)
    an = np.multiply(an,an)
    print(sqrt(an.sum()))
    # End

    a = GaussianRandomProjection(n_components=10, random_state=0).fit_transform(X_train)

    x = range(1,21, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('cluster distance')
    for i in x:
        g = GaussianMixture(n_components=i, random_state=0,init_params='random_from_data').fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.means_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist =  tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    x = range(1, 21, 3)
    z = []
    plt.xlabel('clusters')
    plt.ylabel('cluster distance')
    for i in x:
        g = KMeans(n_clusters=i, random_state=0).fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.cluster_centers_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist = tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    g = GaussianMixture(n_components=13, random_state=0, init_params='random_from_data').fit(a)
    print('EM')
    print(homogeneity_score(y_train, g.predict(a)))

    g = KMeans(n_clusters=10, random_state=0).fit(a)
    print('KMeans')
    print(homogeneity_score(y_train, g.predict(a)))

    print('NNReduction')

    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                        cv=StratifiedKFold(n_splits=5), n_jobs=10, scoring='f1_micro')
    vis.fit(a, np.ravel(y_train))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                          param_name="max_iter",
                          param_range=np.arange(0, 210, 10), cv=StratifiedKFold(n_splits=5), scoring='f1_micro',
                          n_jobs=10)

    vis.fit(a, np.ravel(y_train))
    vis.show()
    # end

    v = MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95))

    strt = time.time()
    v.fit(a, y_train)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(a)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, average='micro'))
    print(' ')

def ica1():
    data = pd.read_csv('./adult/adult.data')
    X = data.iloc[:10000, :14]
    X = OneHotEncoder(handle_unknown='ignore').fit_transform(X)
    y = data.iloc[:10000, -1]

    x = range(1,31, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('kurtosis')
    for i in x:
        print(i)
        g = FastICA(n_components=i, random_state=0).fit_transform(X.toarray())
        temp = scipy.stats.kurtosis(g)
        z.append(sum(temp)/len(temp))

    plt.plot(x, z)
    plt.show()

    a = FastICA(n_components=16, random_state=0).fit(X.toarray())
    # code from https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn, metric from https://core.ac.uk/download/pdf/148661309.pdf
    g = a.transform(X.toarray())
    x_rec = a.inverse_transform(g)
    an = (X - x_rec)
    an = np.multiply(an,an)
    print(sqrt(an.sum()))
    # End

    a = FastICA(n_components=16, random_state=0).fit_transform(X.toarray())
    x = range(1,21, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('cluster distance')
    for i in x:
        g = GaussianMixture(n_components=i, random_state=0,init_params='random_from_data').fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.means_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist =  tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    x = range(1, 21, 3)
    z = []
    plt.xlabel('clusters')
    plt.ylabel('cluster distance')
    for i in x:
        g = KMeans(n_clusters=i, random_state=0).fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.cluster_centers_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist = tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    g = GaussianMixture(n_components=16, random_state=0, init_params='random_from_data').fit(a)
    print('EM')
    print(homogeneity_score(y.squeeze(), g.predict(a)))

    g = KMeans(n_clusters=12, random_state=0).fit(a)
    print('KMeans')
    print(homogeneity_score(y.squeeze(), g.predict(a)))

def ica2():
    # code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    # end
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)
    # end

    x = range(1,31, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('kurtosis')
    for i in x:
        print(i)
        g = FastICA(n_components=i, random_state=0).fit_transform(X_train)
        temp = scipy.stats.kurtosis(g)
        z.append(sum(temp)/len(temp))

    plt.plot(x, z)
    plt.show()

    a = FastICA(n_components=13, random_state=0).fit(X_train)
    # code from https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn, metric from https://core.ac.uk/download/pdf/148661309.pdf
    g = a.transform(X_train)
    x_rec = a.inverse_transform(g)
    an = (X_train - x_rec)
    an = np.multiply(an,an)
    print(sqrt(an.sum()))
    # End

    a = FastICA(n_components=13, random_state=0).fit_transform(X_train)
    x = range(1,21, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('cluster distance')
    for i in x:
        g = GaussianMixture(n_components=i, random_state=0,init_params='random_from_data').fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.means_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist =  tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    x = range(1, 21, 3)
    z = []
    plt.xlabel('clusters')
    plt.ylabel('cluster distance')
    for i in x:
        g = KMeans(n_clusters=i, random_state=0).fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.cluster_centers_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist = tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    g = GaussianMixture(n_components=11, random_state=0, init_params='random_from_data').fit(a)
    print('EM')
    print(homogeneity_score(y_train, g.predict(a)))

    g = KMeans(n_clusters=11, random_state=0).fit(a)
    print('KMeans')
    print(homogeneity_score(y_train, g.predict(a)))

    print('NNReduction')

    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                        cv=StratifiedKFold(n_splits=5), n_jobs=10, scoring='f1_micro')
    vis.fit(a, np.ravel(y_train))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                          param_name="max_iter",
                          param_range=np.arange(0, 210, 10), cv=StratifiedKFold(n_splits=5), scoring='f1_micro',
                          n_jobs=10)

    vis.fit(a, np.ravel(y_train))
    vis.show()
    # end

    v = MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95))

    strt = time.time()
    v.fit(a, y_train)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(a)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, average='micro'))
    print(' ')

def pca1():
    data = pd.read_csv('./adult/adult.data')
    X = data.iloc[:10000, :14]
    X = OneHotEncoder(handle_unknown='ignore').fit_transform(X)
    y = data.iloc[:10000, -1]

    x = range(1,31, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('explained_variance')
    for i in x:
        print(i)
        g = PCA(n_components=i, random_state=0).fit(X.toarray())

        z.append(sum(g.explained_variance_)/len(g.explained_variance_))

    plt.plot(x, z)
    plt.show()

    a = PCA(n_components=10, random_state=0).fit(X.toarray())
    # code from https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn, metric from https://core.ac.uk/download/pdf/148661309.pdf
    g = a.transform(X.toarray())
    x_rec = a.inverse_transform(g)
    an = (X - x_rec)
    an = np.multiply(an,an)
    print(sqrt(an.sum()))
    # End

    a = PCA(n_components=10, random_state=0).fit_transform(X.toarray())

    x = range(1,21, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('cluster distance')
    for i in x:
        g = GaussianMixture(n_components=i, random_state=0,init_params='random_from_data').fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.means_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist =  tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    x = range(1, 21, 3)
    z = []
    plt.xlabel('clusters')
    plt.ylabel('cluster distance')
    for i in x:
        g = KMeans(n_clusters=i, random_state=0).fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.cluster_centers_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist = tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    g = GaussianMixture(n_components=10, random_state=0, init_params='random_from_data').fit(a)
    print('EM')
    print(homogeneity_score(y.squeeze(), g.predict(a)))

    g = KMeans(n_clusters=10, random_state=0).fit(a)
    print('KMeans')
    print(homogeneity_score(y.squeeze(), g.predict(a)))

def pca2():
    # code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas"
    )
    # end
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)
    # end

    x = range(1,31, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('explained_variance')
    for i in x:
        print(i)
        g = PCA(n_components=i, random_state=0).fit(X_train)

        z.append(sum(g.explained_variance_)/len(g.explained_variance_))

    plt.plot(x, z)
    plt.show()

    a = PCA(n_components=12, random_state=0).fit(X_train)
    # code from https://stackoverflow.com/questions/36566844/pca-projection-and-reconstruction-in-scikit-learn, metric from https://core.ac.uk/download/pdf/148661309.pdf
    g = a.transform(X_train)
    x_rec = a.inverse_transform(g)
    an = (X_train - x_rec)
    an = np.multiply(an,an)
    print(sqrt(an.sum()))
    # End

    a = PCA(n_components=12, random_state=0).fit_transform(X_train)

    x = range(1, 21, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('cluster distance')
    for i in x:
        g = GaussianMixture(n_components=i, random_state=0, init_params='random_from_data').fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.means_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist = tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    x = range(1, 21, 3)
    z = []
    plt.xlabel('clusters')
    plt.ylabel('cluster distance')
    for i in x:
        g = KMeans(n_clusters=i, random_state=0).fit(a)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.cluster_centers_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist = tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    g = GaussianMixture(n_components=13, random_state=0, init_params='random_from_data').fit(a)
    print('EM')
    print(homogeneity_score(y_train, g.predict(a)))

    g = KMeans(n_clusters=10, random_state=0).fit(a)
    print('KMeans')
    print(homogeneity_score(y_train, g.predict(a)))

    print('NNReduction')

    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                        cv=StratifiedKFold(n_splits=5), n_jobs=10, scoring='f1_micro')
    vis.fit(a, np.ravel(y_train))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                          param_name="max_iter",
                          param_range=np.arange(0, 210, 10), cv=StratifiedKFold(n_splits=5), scoring='f1_micro',
                          n_jobs=10)

    vis.fit(a, np.ravel(y_train))
    vis.show()
    # end

    v = MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95))

    strt = time.time()
    v.fit(a, y_train)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(a)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, average='micro'))
    print(' ')


def clustering1():
    data = pd.read_csv('./adult/adult.data')
    X = data.iloc[:10000, :14]
    X = OneHotEncoder(handle_unknown='ignore').fit_transform(X)
    y = data.iloc[:10000, -1]

    x = range(1,21, 3)
    z = []
    plt.xlabel('components')
    plt.ylabel('cluster distance')
    for i in x:
        g = GaussianMixture(n_components=i, random_state=0,init_params='random_from_data').fit(X.toarray())
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.means_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist =  tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    g = GaussianMixture(n_components=13, random_state=0, init_params='random_from_data').fit(X.toarray())
    print(y)
    print(homogeneity_score(y.squeeze(), g.predict(X.toarray())))

    x = range(1, 21, 3)
    z = []
    plt.xlabel('clusters')
    plt.ylabel('cluster distance')
    for i in x:
        g = KMeans(n_clusters=i, random_state=0).fit(X.toarray())
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.cluster_centers_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist = tri_dists.mean()
        print(avg_dist)
        z.append(avg_dist)
        # end
    plt.plot(x, z)
    plt.show()

    g = KMeans(n_clusters=8, random_state=0).fit(X.toarray())
    print(homogeneity_score(y.squeeze(), g.predict(X.toarray())))

def clustering2():
    # code from https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mnist_filters.html
    X, y = fetch_openml(
        "mnist_784", version=1, return_X_y=True, as_frame=True, parser="pandas"
    )
    # end
    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)
    # end

    x = range(1,21, 3)
    y = []
    plt.xlabel('components')
    plt.ylabel('cluster distance')
    for i in x:
        g = GaussianMixture(n_components=i, random_state=0,init_params='random_from_data').fit(X_train)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.means_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist =  tri_dists.mean()
        print(avg_dist)
        y.append(avg_dist)
        # end
    plt.plot(x, y)
    plt.show()

    g = GaussianMixture(n_components=10, random_state=0, init_params='random_from_data').fit(X_train)

    print(homogeneity_score(y_train, g.predict(X_train)))

    # code from https://datascience.stackexchange.com/questions/47264/how-to-create-new-feature-based-on-clustering-result
    X_train['clusters'] = g.predict(X_train)
    X_test['clusters'] = g.predict(X_test)
    # end


    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                        cv=StratifiedKFold(n_splits=5), n_jobs=10, scoring='f1_micro')
    vis.fit(X_train, np.ravel(y_train))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                          param_name="max_iter",
                          param_range=np.arange(0, 210, 10), cv=StratifiedKFold(n_splits=5), scoring='f1_micro',
                          n_jobs=10)

    vis.fit(X_train, np.ravel(y_train))
    vis.show()
    # end

    v = MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95))

    strt = time.time()
    v.fit(X_train, y_train)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, average='micro'))
    print(' ')

    # code modified from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)
    # end

    x = range(1, 21, 3)
    y = []
    plt.xlabel('clusters')
    plt.ylabel('cluster distance')
    for i in x:
        g = KMeans(n_clusters=i, random_state=0).fit(X_train)
        # code from https://stackoverflow.com/questions/51729851/distance-between-clusters-kmeans-sklearn-python
        dists = euclidean_distances(g.cluster_centers_)
        tri_dists = dists[np.triu_indices(i, 1)]
        avg_dist = tri_dists.mean()
        print(avg_dist)
        y.append(avg_dist)
        # end
    plt.plot(x, y)
    plt.show()

    g = KMeans(n_clusters=9, random_state=0).fit(X_train)

    print(homogeneity_score(y_train, g.predict(X_train)))

    # code from https://datascience.stackexchange.com/questions/47264/how-to-create-new-feature-based-on-clustering-result
    X_train['clusters'] = g.labels_
    X_test['clusters'] = g.predict(X_test)
    # end

    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/learning_curve.html
    vis = LearningCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                        cv=StratifiedKFold(n_splits=5), n_jobs=10, scoring='f1_micro')
    vis.fit(X_train, np.ravel(y_train))
    vis.show()
    # end
    # code modified from https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
    vis = ValidationCurve(MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95)),
                          param_name="max_iter",
                          param_range=np.arange(0, 210, 10), cv=StratifiedKFold(n_splits=5), scoring='f1_micro',
                          n_jobs=10)

    vis.fit(X_train, np.ravel(y_train))
    vis.show()
    # end

    v = MLPClassifier(random_state=0, batch_size=50, hidden_layer_sizes=(95, 95))

    strt = time.time()
    v.fit(X_train, y_train)
    ned = time.time()
    print('training Time: ' + str(ned - strt) + 's')

    strt = time.time()
    ans = v.predict(X_test)
    ned = time.time()
    print('inference Time: ' + str(ned - strt) + 's')

    print('score:')
    print(f1_score(y_test, ans, average='micro'))
    print(' ')




if __name__ == '__main__':
    main()


