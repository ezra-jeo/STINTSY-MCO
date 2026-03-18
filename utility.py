import re
import pandas as pd
import numpy as np
import emoji
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd

class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms text + linguistic features into combined feature matrix.
    
    Usage:
        transformer = FeatureTransformer(text_col='text_traditional')
        transformer.fit(X_train)
        X_train_transformed = transformer.transform(X_train)
        X_test_transformed = transformer.transform(X_test)
    """
    
    def __init__(self, text_col='text_traditional', ngram_range=(1, 2), max_features=10000):
        self.text_col = text_col
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        self.scaler = StandardScaler()
        self.feature_cols = None  # Will be set during fit
    
    def fit(self, X, y=None):
        """
        Fit TF-IDF and StandardScaler on training data.
        
        
        :param X: DataFrame with text column and feature columns
        :param y: Ignored (for sklearn compatibility)
        """
        # Identify feature columns (everything except text column)
        self.feature_cols = X.columns[X.columns != self.text_col]
        
        # Fit TF-IDF on text
        self.tfidf.fit(X[self.text_col])
        
        # Fit scaler on linguistic features
        if len(self.feature_cols) > 0:
            self.scaler.fit(X[self.feature_cols])
        
        return self 
    
    def transform(self, X):
        """
        Transform data into combined feature matrix.
        
        
        :param X: DataFrame with text column and feature columns
            
        :return Sparse matrix: hstack([TF-IDF features, scaled linguistic features])
        """
        # Transform text with TF-IDF (sparse matrix)
        X_tfidf = self.tfidf.transform(X[self.text_col])
        
        # Transform linguistic features with scaler (dense array)
        if len(self.feature_cols) > 0:
            X_features = self.scaler.transform(X[self.feature_cols])
            
            # Combine sparse TF-IDF + dense features
            X_features_sparse = csr_matrix(X_features)
            X_combined = hstack([X_tfidf, X_features_sparse])
        else:
            # No linguistic features, just return TF-IDF
            X_combined = X_tfidf
        
        return X_combined
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one function.

        :param X: DataFrame with text column and feature columns
        :param y: Ignored (for sklearn compatibility)
        """
        return self.fit(X, y).transform(X)