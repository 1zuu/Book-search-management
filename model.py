import os
import json

from sklearn.metrics.pairwise import cosine_similarity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import pathlib
import warnings
import numpy as np
import scipy.stats as st
import tensorflow as tf
from matplotlib import pyplot as plt
logging.getLogger('tensorflow').disabled = True
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam

from variables import *
from util import*

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\n Num GPUs Available: {}\n".format(len(physical_devices)))
if len(physical_devices):
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

np.random.seed(seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
warnings.simplefilter("ignore", DeprecationWarning)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.99):
            print("\nReached 99% train accuracy.So stop training!")
            self.model.stop_training = True

class BSM_Model(object):
    def __init__(self):
        X, Y,df_response = get_Data()
        self.size_output = len(set(Y))

        # X, Xtest, Y, Ytest = train_test_split(
        #                                     X, Y, 
        #                                     test_size=test_size, 
        #                                     random_state=seed
        #                                     )
        self.X = X
        self.Y = Y
        self.df_response = df_response
        # self.Xtest = Xtest
        # self.Ytest = Ytest

    def length_analysis(self, X_seq, X_seq_test):
        len_x = [len(sen) for sen in X_seq]
        len_xtest = [len(sen) for sen in X_seq_test]
        Xlen = np.array(len_x + len_xtest)
        q25, q75 = np.percentile(Xlen,[.25,.75])
        bin_width = 2*(q75 - q25)*len(Xlen)**(-1/3)
        bins = int(round((Xlen.max() - Xlen.min())/bin_width))

        plt.hist(Xlen, density=True, bins=bins, label="Text Lengths")
        mn, mx = plt.xlim()
        plt.xlim(mn, mx)
        kde_xs = np.linspace(mn, mx, 300)
        kde = st.gaussian_kde(Xlen)
        plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
        plt.legend(loc="upper left")
        plt.ylabel('occurance')
        plt.xlabel('lengths')
        plt.title("Histogram")
        plt.show()

    def handle_data(self):
        if not os.path.exists(model_weights):
            tokenizer = Tokenizer(
                            oov_token=oov_tok
                                ) # Create Tokenizer Object
            tokenizer.fit_on_texts(self.X) # Fit tokenizer with train data
            tokenizer_save_and_load(tokenizer)
        
        else:
            tokenizer = tokenizer_save_and_load()     

        X_seq = tokenizer.texts_to_sequences(self.X) # tokenize train data
        self.X_pad = pad_sequences(X_seq, maxlen=max_length, padding=padding, truncating=trunc_type)# Pad Train data

        # X_seq_test = tokenizer.texts_to_sequences(self.Xtest) # tokenize train data
        # self.X_pad_test = pad_sequences(X_seq_test, maxlen=max_length, padding=padding, truncating=trunc_type )# Pad Train data
        X_seq_test = []
        self.length_analysis(X_seq, X_seq_test)
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.word_index) + 1

    def subnetwork(dense, x):
        # x = Dense(dense, activation='relu')(x)
        x = Dense(dense)(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(rate)(x)
        return x
        
    def feature_extractor(self): # Building the RNN model
        inputs = Input(shape=(max_length,), name='text_inputs')
        x = Embedding(output_dim=embedding_dim, input_dim=self.vocab_size, input_length=max_length, name='embedding')(inputs) # Embedding layer
        x = Bidirectional(LSTM(size_lstm,return_sequences=True,unroll=True), name='lstm1')(x)
        x = Bidirectional(LSTM(size_lstm // 2,unroll=True), name='lstm2')(x)

        x = BSM_Model.subnetwork(dense1, x)
        x = BSM_Model.subnetwork(dense2, x)
        x = BSM_Model.subnetwork(dense3, x)
        # x = BSM_Model.subnetwork(dense4, x)

        x = Dense(dense4, activation='relu', name='text_features')(x)
        outputs = Dense(self.size_output, activation='softmax', name='book_category')(x)

        model = Model(inputs=inputs, outputs=outputs)
        
        self.model = model
        self.model.summary()

    def train(self): # Compile the model and training
        callbacks = myCallback()

        self.model.compile(
                        loss='sparse_categorical_crossentropy', 
                        optimizer=Adam(lr=learning_rate), 
                        metrics=['accuracy']
                        )
        self.history = self.model.fit(
                                self.X_pad,
                                self.Y,
                                # validation_split = 0.15,
                                # batch_size=batch_size,
                                epochs=num_epochs,
                                callbacks=[callbacks]
                                )

    def save_model(self): # Save trained model
        self.feature_model.save(model_weights)

    def load_model(self): # Load and compile pretrained model
        self.feature_model = load_model(model_weights)
        self.feature_model.compile(
                        loss='sparse_categorical_crossentropy', 
                        optimizer=Adam(lr=learning_rate), 
                        metrics=['accuracy']
                        )
        self.feature_model.summary()

    def book_features(self):
        inputs = self.model.input
        outputs = [
                self.model.layers[-1].output, 
                self.model.layers[-2].output
                  ]
        self.feature_model = Model(inputs=inputs, outputs=outputs)
        self.feature_model.summary()

    def TFconverter(self):
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.feature_model)
        converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(model_weights) 

        # converter = tf.lite.TFLiteConverter.from_keras_model(self.feature_model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        model_converter_file = pathlib.Path(tflite_weights)
        model_converter_file.write_bytes(tflite_model)

    def TFinterpreter(self):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=tflite_weights)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def TFliteInference(self, input_data):
        input_shape = self.input_details[0]['shape']
        if len(input_shape) == 1:
            input_data = np.expand_dims(description, axis=0).astype(np.float32)
        assert np.array_equal(input_shape, input_data.shape), "required shape : {} doesn't match with provided shape : {}".format(input_shape, input_data.shape)

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        concern_type = self.interpreter.get_tensor(self.output_details[0]['index']) #Concern_Type
        department = self.interpreter.get_tensor(self.output_details[1]['index']) #Department
            
        return concern_type, department

    def get_category_from_label(self, category):
        with open(encoder_path, 'rb') as handle:
            encoder = pickle.load(handle)
        category = encoder.inverse_transform([category]).squeeze()
        category = str(category).lower()
        return category

    def predict_category(self, description):
        processed_description = prediction_data(description).squeeze()
        category, feature = self.TFliteInference(processed_description.reshape(1,max_length).astype(np.float32))

        category = np.argmax(category)
        category = self.get_category_from_label(category)
        return category, feature
        
    def predict_features(self, category):
        df_response = load_category_df(self.df_response, category)
        df_response = df_response.reset_index(drop=True)
        category_descriptions = df_response['Description'].values
        processed_descriptions = preprocessed_data(category_descriptions)

        features = np.array(
                    [self.TFliteInference(prediction_data(x).reshape(1,max_length).astype(np.float32))[1] for x in processed_descriptions])

        return features, df_response

    def predict_book(self,request):
        description = request['description']
        category, pred_feature = self.predict_category(description)
        features, df_response = self.predict_features(category)
        cos_sim = {i:float(cosine_similarity(pred_feature, feature).squeeze()) for i,feature in enumerate(features)}
        cos_sim = dict(sorted(cos_sim.items(), key=lambda item: item[1], reverse=True))
        top_matches = list(cos_sim.keys())[:n_matches]
        df_top_match = df_response.iloc[top_matches]

        df_top_match = df_top_match[['Book_title' ,'Author' ,'ISBN-10' ,'ISBN-13', 'Cover_link']]
        
        
        response = {}
        books = {}

        response['category'] = category

        for i in range(n_matches):
            book = {}
            book['title'] = df_top_match.iloc[i]['Book_title']
            book['author'] = df_top_match.iloc[i]['Author']
            book['isbn 10'] = df_top_match.iloc[i]['ISBN-10']
            book['isbn 13'] = df_top_match.iloc[i]['ISBN-13']
            book['cover photo'] = df_top_match.iloc[i]['Cover_link']
            books['book {}'.format(i+1)] = book
        response['books'] = books
        return json.dumps(response)

    def run(self):
        self.handle_data()
        if not os.path.exists(tflite_weights):
            if os.path.exists(model_weights):
                self.load_model()
            else:
                self.feature_extractor()
                self.train()
                self.book_features()
                self.save_model()
            self.TFconverter()
        self.TFinterpreter()