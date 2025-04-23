# NLP-sentiment-analysis

This project has been done for the **Machine Learning for NLP** course by Christopher Kermorvant at ENSAE. This repository contains the implementation of a sentiment analysis task on movie reviews using various Natural Language Processing (NLP) techniques. 

## Objective

The goal is to classify movie reviews as positive or negative using different machine learning models, including traditional models like TF-IDF with Naïve Bayes, Word2Vec with Support Vector Classifier (SVC), and modern models like RoBERTa for contextual embeddings.

## Models

1. **Baseline Model: TF-IDF + Naïve Bayes**  
   We used the classical TF-IDF vectorizer followed by naïve Bayes classification to establish a baseline for sentiment classification.

2. **Word2Vec + SVC**  
   We explored Word2Vec embeddings combined with a Support Vector Classifier (SVC) to capture semantic relationships in the movie reviews.

3. **RoBERTa (fine-tuned)**  
   We employed the RoBERTa transformer model, fine-tuned for binary sentiment classification. RoBERTa captures contextualized word embeddings, significantly improving performance on complex reviews.

## Data

The dataset used is the IMDB movie reviews dataset, which consists of 50,000 labeled reviews (25,000 positive and 25,000 negative). 

## How to Run

To reproduce the results, run the following notebooks sequentially:

1. `preprocess_data.ipynb` – for data cleaning and formatting  
2. `exploratory_data_analysis.ipynb` – for visualizing and analyzing the dataset  
3. `main.ipynb` – for training and evaluating the models

## Results

The table below summarizes the performance of each model on the sentiment classification task:

| Model                  | Train Accuracy (%) | Test Accuracy (%) |
|------------------------|--------------------|-------------------|
| TF-IDF + Naïve Bayes   | 86.50              | 84.06             |
| Word2Vec + SVC         | 87.74              | 86.73             |
| RoBERTa (fine-tuned)   | 97.76              | 94.87             |

## Conclusion

This project shows the effectiveness of modern transformer-based models like RoBERTa over traditional machine learning models in sentiment classification tasks. By using contextual embeddings, RoBERTa achieves the highest performance, especially in handling nuances like sarcasm and negation.

## Code Architecture

```plaintext
NLP-SENTIMENT-ANALYSIS/

│── data/            
│   ├── processed/
│   │   ├── datasets/
│   │   │   ├── df_test.csv
│   │   │   ├── df_train.csv
│   │   ├── embeddings/
│   │   │   ├── X_train_word2vec_embeddings.pkl
│   ├── raws/aclImdb/
│   │   ├── test/
│   │   ├── train/
│   │   ├── imdb.vocab
│   │   ├── imdbErt.txt
│   │   ├── stop_word_rate.txt
│── notebooks/           
│   ├── exploratory_data_analysis.ipynb  
│   ├── main.ipynb
│   ├── preprocess_data.ipynb                    
│── src/                    
│   ├── models/               
│   │   ├── tf_idf.py       
│   │   ├── word2vec.py
│   │   ├── roberta.py
│   ├── data_processing.py             
│── README.md               
│── .gitignore
```

## Contact

- **Suzie Grondin** (`suzie(dot)grondin(at)ensae(dot)fr`)
- **Marion Chabrol** (`marion(dot)chabrol(at)ensae(dot)fr`)



