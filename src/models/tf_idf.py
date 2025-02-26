import pandas as pd
from typing import List, Optional

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer



class TfidfClassifier:
    def __init__(self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, 
                 train_file_path: Optional[str] = None, test_file_path: Optional[str] = None):
        self.df_train = pd.read_csv(train_file_path) if test_file_path else None
        self.df_test = pd.read_csv(test_file_path) if test_file_path else None
        self.X_train, self.X_val, self.y_train, self.y_val = X_train, X_val, y_train, y_val
        self.results = []
        self.best_config = None

    # def prepare_data(self, test_size=0.2, random_state=42):
    #     X = self.df_train['comment']
    #     y = self.df_train['sentiment']
    #     self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
    #         X, y, test_size=test_size, random_state=random_state
    #     )

    def run_experiments(self, max_features_list: List[int], use_idf_list: List[bool], alpha_list: List[float]):
        self.results = []
        for max_features in max_features_list:
            for use_idf in use_idf_list:
                for alpha in alpha_list:
                    self.run_single_experiment(max_features, use_idf, alpha)

    def run_single_experiment(self, max_features, use_idf, alpha):
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_counts = vectorizer.fit_transform(self.X_train)
        X_val_counts = vectorizer.transform(self.X_val)

        tfidf_transformer = TfidfTransformer(use_idf=use_idf)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        X_val_tfidf = tfidf_transformer.transform(X_val_counts)

        clf = MultinomialNB(alpha=alpha)
        clf.fit(X_train_tfidf, self.y_train)

        train_accuracy = clf.score(X_train_tfidf, self.y_train)
        val_accuracy = clf.score(X_val_tfidf, self.y_val)

        config = {
            'max_features': max_features,
            'use_idf': use_idf,
            'alpha': alpha,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy
        }

        if not hasattr(self, 'results'):
            self.results = []

        self.results.append(config)

    def get_best_config(self):
        self.best_config = max(self.results, key=lambda x: x['val_accuracy'])
        return self.best_config

    def evaluate_on_test(self, config: dict):
        if self.df_test is None or self.df_train is None:
            raise ValueError("No train or test dataset provided")

        vectorizer = CountVectorizer(max_features=config['max_features'])
        X_train_counts = vectorizer.fit_transform(self.df_train['comment'])
        X_test_counts = vectorizer.transform(self.df_test['comment'])

        tfidf_transformer = TfidfTransformer(use_idf=config['use_idf'])
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        X_test_tfidf = tfidf_transformer.transform(X_test_counts)

        clf = MultinomialNB(alpha=config['alpha'])
        clf.fit(X_train_tfidf, self.df_train['sentiment'])

        train_accuracy = clf.score(X_train_tfidf, self.df_train['sentiment'])
        test_accuracy = clf.score(X_test_tfidf, self.df_test['sentiment'])
        return train_accuracy, test_accuracy