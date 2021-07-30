n_dim = 100
max_length = 20
trunc_type = 'post'
padding = 'post'
pad_token = '<pad>'

encoder_path = 'weights_and_data/label_encoder.pickle'
wordcloud_path = 'weights_and_data/wordcloud.png'
data_path = 'weights_and_data/books.csv'

seed = 42
user_name = 'root'
password = 'root'
db_url = 'mysql+pymysql://{}:{}@localhost:3306/sms'
table_name = 'book_management'