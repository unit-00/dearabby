import sys
import os
import pickle
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation
from helper import print_centroid_top_words, print_top_categories, print_top_words
from stemmed_vect import *

def run_models():
    # import data
    # path = (Path(__file__)/'../../data/dearabby_qa.csv').resolve()
    data = pd.read_csv('data/dearabby_qa.csv')
    df = data[data['categories'].notna()]
    X = df['qa']
    y = df['categories']

    # import models
    with open('model/scv.model', 'rb') as scv_file:
        scv = pickle.load(scv_file)
        df_x = scv.transform(X)

    with open('model/stv.model', 'rb') as stv_file:
        stv = pickle.load(stv_file)
        tfidf_x = stv.transform(X)

    with open('model/pca_tv.model', 'rb') as pca_file:
        pca = pickle.load(pca_file)
        pca_x = pca.transform(tfidf_x.toarray())
    
    
    km_clusters = 3
    nmf_components = 4
    lda_components = 4

    km = KMeans(random_state=42, n_clusters=km_clusters, n_jobs=-1).fit(pca_x)
    nmf = NMF(n_components=nmf_components, random_state=42,).fit(tfidf_x)
    lda = LatentDirichletAllocation(n_components=lda_components, 
                                   max_iter=5,
                                   learning_method='online',
                                   learning_offset=50.,
                                   random_state=42,
                                   n_jobs=-1).fit(df_x)

    # KMeans keywords and categories
    print('Kmeans: ', end='\n')
    print('Keywords: ', end='\n')
    get_kmeans_keywords(pca, km, stv, km_clusters)
    print()
    
    print('Categories: ', end='\n')
    km_labels = pd.concat([y, pd.Series(km.labels_, index=y.index)], axis=1)
    print_top_categories(km_labels, km_clusters)
    print()

    # NMF keywords and categories
    print('NMF: ', end='\n')
    n_top_words = 20
    

    print('Keywords: ', end='\n')
    tfidf_feature_names = stv.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words=n_top_words)
    print()

    print('Categories: ', end='\n')
    nmf_labels = pd.concat([y, pd.Series(nmf.transform(tfidf_x).argmax(axis=1), index=y.index)], axis=1)
    print_top_categories(nmf_labels, nmf_components)
    print()

    # LDA keywords and categories
    print('LDA: ', end='\n')
    lda_tf = lda.transform(df_x)

    print('Keywords: ', end='\n')
    tf_feature_names = scv.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
    print()

    print('Categories: ', end='\n')
    lda_labels = pd.concat([y, pd.Series(lda_tf.argmax(axis=1), index=y.index)], axis=1)

    print_top_categories(lda_labels, lda_components)
    print()

    return 

def get_kmeans_keywords(pca, km, stv, n):
    original_space_centroids = pca.inverse_transform(km.cluster_centers_)
    terms = stv.get_feature_names()

    print_centroid_top_words(terms, original_space_centroids, n)

    return

from pathlib import Path

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print('Too many arguments')
        sys.exit(1)

    run_models()