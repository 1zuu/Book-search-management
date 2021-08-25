import os
import re
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.utils import shuffle
from wordcloud import WordCloud
from nltk.corpus import stopwords

from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences

from variables import*

def lemmatization(lemmatizer,sentence):
    '''
        Lemmatize texts in the terms
    '''
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = list(dict.fromkeys(lem))

    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    '''
        Remove stop words in texts in the terms
    '''
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(description):
    '''
        Text preprocess on term text using above functions
    '''
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    description = description.lower()
    remove_punc = tokenizer.tokenize(description) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_punc if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_description = ' '.join(remove_stop)
    return updated_description

def preprocessed_data(descriptions):
    '''
        Preprocess entire terms
    '''
    updated_descriptions = []
    if isinstance(descriptions, np.ndarray) or isinstance(descriptions, list):
        for description in descriptions:
            updated_description = preprocess_one(description)
            updated_descriptions.append(updated_description)
    elif isinstance(descriptions, np.str_)  or isinstance(descriptions, str):
        updated_descriptions = [preprocess_one(descriptions)]

    return np.array(updated_descriptions)

def label_encoding(Y):
    Y = Y.str.strip()
    Y = Y.str.lower()
    if not os.path.exists(encoder_path):
        encoder = LabelEncoder()
        encoder.fit(Y)
        
        with open(encoder_path, 'wb') as handle:
            pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(encoder_path, 'rb') as handle:
            encoder = pickle.load(handle)
    return encoder.transform(Y)

def create_wordcloud(processed_descriptions):
    if not os.path.exists(wordcloud_path):
        long_string = ','.join(list(processed_descriptions))
        wordcloud = WordCloud(
                            width=1600, 
                            height=800, 
                            max_words=200, 
                            background_color='white',
                            max_font_size=200, 
                            random_state=seed
                            )
        wordcloud.generate(long_string)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("WordCloud Distribution of Student descriptions")
        plt.savefig(wordcloud_path)
        plt.show()

def connect_mongo():
    client = MongoClient(db_url)
    db = client[database]
    return db

def create_database():

    db = connect_mongo()
    if db_collection not in db.list_collection_names():
        coll = db[db_collection]
        data = pd.read_csv(data_path)
        data = data.dropna(axis=0)
        payload = json.loads(data.to_json(orient='records'))
        coll.remove()
        coll.insert(payload)
        print('Database created')

def read_mongo():
    db = connect_mongo()
    cursor = db[db_collection].find({})
    df =  pd.DataFrame(list(cursor))
    del df['_id']

    return df

def get_Data():
    '''
        Get data from database
    '''
    create_database()

    df = read_mongo()
    df_response = df[['Category', 'Description', 'Book_title' ,'Author' ,'ISBN-10' ,'ISBN-13', 'Cover_link']] 
    df= df[['Description', 'Category']]
    description = df['Description'].values
    Category = df['Category']

    X = preprocessed_data(description)
    create_wordcloud(X)

    Y = label_encoding(Category)

    X, Y = shuffle(X, Y, random_state=seed) 
    return X, Y, df_response

def tokenizer_save_and_load(tokenizer=None):
    if tokenizer:
        file_ = open(tokenizer_weights,'wb')
        pickle.dump(tokenizer, file_, protocol=pickle.HIGHEST_PROTOCOL)
        file_.close()
    else:
        assert os.path.exists(tokenizer_weights), "Tokenizer Weights doesn't exists. Please Save before load."
        file_ = open(tokenizer_weights,'rb')
        tokenizer = pickle.load(file_)
        file_.close()
        return tokenizer

def prediction_data(description):
    tokenizer = tokenizer_save_and_load()
    processed_description = preprocess_one(description)
    x_seq = tokenizer.texts_to_sequences([processed_description])
    x_pad = pad_sequences(x_seq, maxlen=max_length, padding=padding, truncating=trunc_type)
    return x_pad

def load_prices(df):
    prices = df[['price{}'.format(i) for i in range(1,n_book_stores + 1)]]
    prices = prices.apply(lambda x: x.str.strip())
    prices = prices.apply(lambda x: x.str.replace('$', ''))
    prices = prices.apply(lambda x: x.str.replace(',', '').astype(float)).values
    stores = df[websites].values

def load_category_df(df_response, pred_category):
    df_response['Category'] = df_response['Category'].str.strip()
    df_response['Category'] = df_response['Category'].str.lower()
    df_response = df_response[df_response['Category'] == pred_category]
    return df_response