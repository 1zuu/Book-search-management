n_dim = 100
max_length = 20
trunc_type = 'post'
padding = 'post'
pad_token = '<pad>'

model_weights = 'weights_and_data/model_weights.h5'
encoder_path = 'weights_and_data/label_encoder.pickle'
wordcloud_path = 'weights_and_data/wordcloud.png'
data_path = 'weights_and_data/books.csv'

seed = 42
username = 'root'
password = 'root'
db_url = 'mysql+pymysql://{}:{}@localhost:3306/sms'.format(username,password)
table_name = 'book_management'

vocab_size = 3500
max_length = 30
embedding_dimS = 256
trunc_type = 'post'
oov_tok = "<OOV>"
num_epochs = 20
batch_size = 64
size_lstm  = 256
dense1 = 256
dense2 = 128
dense3 = 64
keep_prob = 0.4

test_size = 0.02