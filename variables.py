max_length = 140
trunc_type = 'post'
padding = 'post'
pad_token = '<pad>'
oov_tok = "<oov>"

model_weights = 'weights/feature_model_weights.h5'
tflite_weights = 'weights/feature_model_weights.tflite'
encoder_path = 'weights/label_encoder.pickle'
tokenizer_weights = 'weights/tokenizer.pickle'
wordcloud_path = 'data/wordcloud.png'
data_path = 'data/books_V2.csv'

seed = 42
host = '127.0.0.1'
port = 27017
database = 'BSMS'
db_collection = 'books'
live_collection = 'live_books'
username = 'root'
password = 'root'
# db_url = 'mongodb://{}:{}@{}:{}/{}?authSource=admin'.format(username, password, host, port, database)
db_url = "mongodb://localhost:27017/"

embedding_dim = 150
num_epochs = 20
batch_size = 32
size_lstm  = 256
dense1 = 512
dense2 = 256
dense3 = 128
dense4 = 64
rate = 0.2
learning_rate = 0.001

test_size = 0.25

n_matches = 5
n_book_stores = 10
websites = [
    'amazon_link', 'ebay_link', 'BetterWorldBooks', 
    'Bernes & Nobel', 'target.com', 'AbeBooks.com',
    'alibris.com', 'strandbooks.com', 'Magers&Quinn.com', 
    'thriftbooks']