import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.manifold import TSNE
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import matplotlib.pyplot as plt
import seaborn as sns


def normalize_data():
    df = pd.read_csv('dataset.csv')

    scaler = preprocessing.MinMaxScaler()
    df_normalizado = scaler.fit_transform(df)
    df_normalizado = pd.DataFrame(df_normalizado, columns=df.columns)  # Hay que convertir a DF el resultado.
    df_normalizado.to_csv('DATA_normalized.csv')

    return df_normalizado

def tsne_plots(k_value):
    dataset = pd.read_csv('predictions/data_cluster' + str(k_value) + '.csv')

    df = pd.DataFrame(dataset)
    x = df.iloc[:, 1:-1].values
    y = df['Cluster'].values

    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", int(k_value)),
                    data=df).set(title='T-SNE Projection k = ' + str(k_value))
    plt.show()


if __name__ == '__main__':

    data = normalize_data()
    model = MiniBatchKMeans()

    inertias = []

    # optimize k [2-12]
    for k in range(2, 13):
        kmeans = MiniBatchKMeans(n_clusters=k)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

        # adds indicator (class-label) to file
        label = kmeans.labels_
        data["Cluster"] = label
        data["Cluster"] = data["Cluster"].astype("int")

        # save each df with different k
        data.to_csv('predictions/data_cluster' + str(k) + '.csv')

    visualizer = KElbowVisualizer(model, k=(2, 13), metric='distortion', timings=False,
                                  title= ('Mini Batch K-Means Clustering'))

    visualizer.fit(data)  # Fit the data to the visualizer
    k_scores = visualizer.k_scores_

    visualizer.show()  # Finalize and render the figure

    print('-- Average Distance to Centroid --')
    for i in range(len(k_scores)):
        # print average distances for each k
        print('K score: ' + str(i+2) + ' ' + str(k_scores[i]))

    best_k = visualizer.elbow_value_
    print('-- Best Value of K: ', best_k , ' --')

    # tsne projection best_k
    tsne_plots(best_k)

    # tsne projection best_k -1
    tsne_plots(best_k-1)

    # tsne projection best_k +1
    tsne_plots(best_k+1)


