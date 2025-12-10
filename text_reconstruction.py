import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras import layers,models
import sys

def build_tfidf(df,text_cols,max_features=20000,min_df=5):
    text = (df[text_cols].fillna('').agg(' '.join,axis=1))
    tfidf = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words='english'
    )

    X = tfidf.fit_transform(text)
    return X,tfidf

def reduce_text(X,n_components=300,seed=42):
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=seed
    )

    X_red = svd.fit_transform(X)
    return X_red,svd

def run_text_autoencoder(X,epochs=100,batch_size=256,seed=42,verbose=0):
    np.random.seed(seed)
    dim = X.shape[1]
    ae = models.Sequential([
        layers.Input(shape=(dim,)),
        layers.Dense(128,activation='relu'),
        layers.Dense(64,activation='relu'),
        layers.Dense(32,activation='relu'),
        layers.Dense(16,activation='relu'),
        layers.Dense(32,activation='relu'),
        layers.Dense(64,activation='relu'),
        layers.Dense(128,activation='relu'),
        layers.Dense(dim)
    ])

    ae.compile(optimizer='adam',loss='mse')
    ae.fit(X,X,
           epochs=epochs,
           batch_size=batch_size,
           verbose=verbose)
    X_rec = ae.predict(X,verbose=0)
    recon_error = np.mean((X - X_rec)**2,axis=1)

    return recon_error,ae


if __name__ == "__main__":
    df = pd.read_csv("fake_job_postings.csv")

    TEXT_COLS = ['title','description','requirements','company_profile','benefits']
    print(f'Loaded data: {df.shape}')

    X_tfidf,tfidf = build_tfidf(df,TEXT_COLS)
    print(f'TF-IDF shape: {X_tfidf.shape}')

    X_svd,svd = reduce_text(X_tfidf)
    print(f'SVD shape: {X_svd.shape}, variance={svd.explained_variance_ratio_.sum():.3f}')

    recon_err,_ = run_text_autoencoder(X_svd)
    print('Reconstruction error summary:')
    print(pd.Series(recon_err).describe())

