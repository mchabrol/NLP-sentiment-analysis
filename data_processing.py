# basic imports
import os
import re
import numpy as np
import pandas as pd

#Code
from typing import List

# display matplotlib graphics in notebook
#%matplotlib inline 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# disable warnings for libraries
import warnings
warnings.filterwarnings("ignore")

# configure logger
import logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)


class EmbeddingLoader:
    def __init__(self, rates_path, words_path):
        self.rates_path = rates_path
        self.words_path = words_path
        self.words = []
        self.rates = []
        self.stopwords = []

    def load_embeddings(self):
        """
        load words and corresponding embedding vectors and check if dimensions match
        """
        logger.info("Loading embedding vectors...")
        self.rates= np.genfromtxt('aclImdb/imdbEr.txt')
        logger.info(f"Loaded vectors with shape: {self.rates.shape}")

        logger.info("Loading words...")
        with open(self.words_path, 'r', encoding='utf-8') as f:
            self.words = [line.strip() for line in f.readlines()]
        logger.info(f"Loaded {len(self.words)} words")

        if len(self.words) != self.rates.shape[0]:
            raise ValueError(f"Mismatch between number of words ({len(self.words)}) and vectors ({self.rates.shape[0]})")

        logger.info(f"Loaded {len(self.words)} words and embeddings.")

    def save_stopwords(self, filepath: str, treshold=0.7) -> list:
        """get and save stopwords (words with occurrence > threshold)
        Args 
            filepath : where to save the stopwords file
        Return 
            a list of stopwords
        """
        self.stopwords = [word for word, rate in zip(self.words, self.rates) if np.abs(rate) < treshold]
        print(f"Nombre de stop words détectés : {len(self.stopwords)}")
        with open(filepath, 'w') as f:
            for word in self.stopwords:
                f.write(word + "\n")
        logger.info(f"Stopwords saved successfully to {filepath}")
        return self.stopwords



class MovieReviewsProcessor:
    def __init__(self, directory_path):
        self.directory = directory_path
        self.df = pd.DataFrame()

    @staticmethod
    def clean_review(text):
        """Removes <br /> """
        return re.sub(r'<br\s*/?>', ' ', text)

    def load_reviews(self):
        """load the reviews and put it in a dataframe"""
        movie_ids, rates, comments = [], [], []
        for filename in os.listdir(self.directory):
            if filename.endswith('.txt'):
                parts = filename.split('_')
                if len(parts) == 2:
                    movie_id, rate = int(parts[0]), int(parts[1].split('.')[0])
                    with open(os.path.join(self.directory, filename), 'r', encoding='utf-8') as file:
                        comment = self.clean_review(file.read())

                        movie_ids.append(movie_id)
                        rates.append(rate)
                        comments.append(comment)

        df = pd.DataFrame({'id': movie_ids, 'rate': rates, 'comment': comments})
        return df.sort_values('id').reset_index(drop=True)
    
    @classmethod
    def create_dataset(cls, pos_reviews_path: str, neg_reviews_path: str, save_path:str) -> pd.DataFrame:
        """create a dataset with positive and negative reviews
        Args 
            pos_reviews_path: path for the positive reviews file
            neg_reviews_path: path for the negative reviews file
        Return
            save_path: directory where the dataset will be saved
        """
        pos_review_loader = cls(directory_path=pos_reviews_path)
        neg_review_loader = cls(directory_path=neg_reviews_path)

        pos_df = pos_review_loader.load_reviews()
        neg_df = neg_review_loader.load_reviews()

        # Labelling
        pos_df['sentiment'] = 1
        neg_df['sentiment'] = 0

        # Create dataset with positive and negative reviews
        df = pd.concat([pos_df, neg_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(save_path)

        return df
    

class DataVisualizer:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def generate_wordcloud(self, stopwords: List[str], col:str):
        categories = sorted(self.df[col].unique())
        for cat in categories:
            text = " ".join(self.df[self.df[col] == cat]['comment'])
            wordcloud = WordCloud(stopwords=stopwords, background_color='white', width=800, height=400).generate(text)
            plt.figure(figsize=(8, 4))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f"WordCloud for {col} {cat}")
            plt.axis('off')
            plt.show()

    def visualize_word_occurrence(self, word: str):
        rates = sorted(self.df['rate'].unique())
        percentages = []

        for rate in rates:
            comments = " ".join(self.df[self.df['rate'] == rate]['comment']).lower().split()
            percent = comments.count(word) / len(comments) * 100
            percentages.append(percent)

        plt.bar(rates, percentages)
        plt.xlabel('Rate')
        plt.ylabel(f"% occurrences of '{word}'")
        plt.title(f"Percentage of occurrences of '{word}' per rate")
        plt.xticks(rates)
        plt.show()
        