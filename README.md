# NLP-sentiment-analysis

## 🔧 Code Architecture

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
│   ├── 1_exploratory_data_analysis.ipynb  
│   ├── 2_main.ipynb                    
│── src/                    
│   ├── models/               
│   │   ├── tf_idf.py       
│   │   ├── word2vec.py
│   ├── data_processing.py             
│── README.md               
│── .gitignore
```
