import os
import re
import numpy as np
import pickle
from typing import List, Tuple
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
logger = logging.getLogger(__name__)


class ReviewTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Basic text cleaning and tokenization"""
        return re.findall(r'\b\w+\b', text.lower())


class Word2VecEmbedder:
    def __init__(self, vector_size: int=100, window:int=10, min_count:int=2, sg:int=1, workers:int=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.workers = workers
        self.model = None
        

    def train(self, tokenized_reviews: List[str]) -> None:
        """Train the Word2Vec model on tokenized reviews
        Args 
            tokenized_reviews: reviews cleaned and tokenized with ReviewTokenizer
        """
        logger.info("Training Word2Vec model...")
        self.model = Word2Vec(
            sentences=tokenized_reviews,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=self.workers
        )
        logger.info("Word2Vec model trained successfully")
        self.wv = self.model.wv

    def embed_reviews(self, tokenized_reviews: List[str]) -> np.ndarray:
        """Compute embeddings by averaging word vectors for each review
        Args 
            tokenized_reviews: List of reviews, each review is a list of tokenized words
        Returns
            np.ndarray: 2D array containing embeddings for each review
        """
        embeddings = []
        for tokens in tokenized_reviews:
            vectors = [self.wv[token] for token in tokens if token in self.wv]
            if vectors:
                embed = np.mean(vectors, axis=0)
            else:
                embed = np.zeros(self.vector_size)
            embeddings.append(embed)
        return np.array(embeddings)

    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> None:
        """Saves the embeddings obtained with word2vec to a given filepath
        Args 
            embeddings: Array containing the embeddings.
            filepath: Path where embeddings will be saved
        """
        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"Embeddings saved successfully to {filepath}")

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Retrieves embeddings obtained with word2vec
        Args
            filepath: Path from where embeddings will be loaded
        Returns 
            np.ndarray: Loaded embeddings array
        """
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings


class SentimentClassifier:
    def __init__(self, classifier=None):
        self.classifier = classifier if classifier else LogisticRegression(max_iter=500)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to train data
        Args
            X: Feature 
            y: True labels 
        """
        self.classifier.fit(X, y)
        logger.info("Classifier trained successfully.")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
        """Evaluate the model on the provided dataset
        Args
            X: Feature 
            y: True labels 
        
        Returns
            Tuple[float, str]: Accuracy score and detailed classification report
        """
        predictions = self.classifier.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        return accuracy, report