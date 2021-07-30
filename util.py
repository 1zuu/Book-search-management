import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import sqlalchemy
import numpy as np
import pandas as pd

from wordcloud import WordCloud
from nltk.corpus import stopwords
from sqlalchemy import create_engine

from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import LabelEncoder

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

def preprocess_one(concern):
    '''
        Text preprocess on term text using above functions
    '''
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    concern = concern.lower()
    remove_punc = tokenizer.tokenize(concern) # Remove puntuations
    remove_num = [i for i in remove_punc if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_concern = ' '.join(remove_stop)
    return updated_concern

def preprocessed_data(concerns):
    '''
        Preprocess entire terms
    '''
    updated_concerns = []
    if isinstance(concerns, np.ndarray) or isinstance(concerns, list):
        for concern in concerns:
            updated_concern = preprocess_one(concern)
            updated_concerns.append(updated_concern)
    elif isinstance(concerns, np.str_)  or isinstance(concerns, str):
        updated_concerns = [preprocess_one(concerns)]

    return np.array(updated_concerns)

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

def create_wordcloud(processed_concerns):
    long_string = ','.join(list(processed_concerns))
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
    plt.title("WordCloud Distribution of Student Concerns")
    plt.savefig(wordcloud_path)
    plt.show()

def create_database(engine):
    engine = create_engine(db_url)
    if table_name not in sqlalchemy.inspect(engine).get_table_names():
        data = pd.read_csv(data_path)
        data = data.dropna(axis=0)
        with engine.connect() as conn, conn.begin():
            data.to_sql(table_name, conn, if_exists='append', index=False)

def get_Data():
    '''
        Get data from database
    '''
    engine = create_engine(db_url.format(user_name,password))
    create_database(engine)

    df = pd.read_sql_table(table_name, engine)
    df= df[['Description', 'Category']]

    description = df['Description'].values
    Category = df['Category']

    X = preprocessed_data(description)
    create_wordcloud(X)

    Y = label_encoding(Category)
    return X, Y