import pickle

class InputPipeline:
    def __init__(self):
        self.is_fitted = False

    def fit(self, text):
        self.text = text

        with open('scv.model', 'rb') as f:
            self.scv = pickle.load(f)

        with open('stv.model', 'rb') as f:
            self.stv = pickle.load(f)

        with open('pca_tv.model', 'rb') as f:
            self.pca_tv = pickle.load(f)

        with open('kmeans.model', 'rb') as f:
            self.kmeans = pickle.load(f)

        with open('nmf.model', 'rb') as f:
            self.nmf = pickle.load(f)

        with open('lda.model', 'rb') as f:
            self.lda = pickle.load(f)

        self.is_fitted = True

        return self

    def transform(self, text):
        if not self.is_fitted:
            raise NotFittedError()
        
        self.df = self.scv.transform(text)
        self.tfidf = self.stv.transform(text)
        self.pca_text = self.pca_tv.transform(self.tfidf)

        kmeans_labels = self.kmeans.predict(self.pca_text)
        nmf_labels = self.nmf.transform(self.tfidf)
        lda_labels = self.lda.transform(self.df)

        return kmeans_labels, nmf_labels, lda_labels

    def fit_transform(self, text):
        self.fit(text)

        return self.transform(text)

class NotFittedError(Exception):
    # Exception raised when pipeline is not fitted

    def __init__(self, message='Pipeline is not fitted yet.'):
        self.message = message

        super().__init__(message)