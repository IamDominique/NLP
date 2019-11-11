import os
import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as k
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, TimeDistributed, Flatten
from bs4 import BeautifulSoup
import re
import nltk
nltk.download("stopwords")   # download list of stopwords
from nltk.corpus import stopwords # import stopwords

class Sentiment_Analysis:

    def load_data(self, directory="./data/imdb-reviews"):
        """Read data a from given directory.

        Directory structure expected:
        - data/
            - train/
                - pos/
                - neg/
            - test/
                - pos/
                - neg/
    """

        # returning data and labels in nested dictionaries matching the directory structure
        self.data = {}
        self.labels = {}

        # sub-directories for the features: train, test
        for data_type in ['train', 'test']:
            self.data[data_type] = {}
            self.labels[data_type] = {}

            #sub-directories for the sentiments (label): pos, neg
            for sentiment in ['pos', 'neg']:
                self.data[data_type][sentiment] = []
                self.labels[data_type][sentiment] = []

                # Fetching the list of files for this sentiment
                path = os.path.join(directory, data_type, sentiment, '*.txt')
                files = glob.glob(path)

                # Read text data and labels and populate dictionaries
                for file in files:
                    with open(file) as text:
                        self.data[data_type][sentiment].append(text.read())
                        self.labels[data_type][sentiment].append(sentiment)

                #checking for discrepencies
                assert len(self.data[data_type][sentiment]) == len(self.labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)



    def join_data(self):
        """
        Prepares training and test sets from the loaded data - combining
        Returns unified training data, test data, training labels and test labels
        """

        #Combinining positive and negative reviews and labels
        self.data_train = self.data['train']['pos'] + self.data['train']['neg']
        self.data_test = self.data['test']['pos'] + self.data['test']['neg']
        self.labels_train = self.labels['train']['pos'] + self.labels['train']['neg']
        self.labels_test = self.labels['test']['pos'] + self.labels['test']['neg']

        #Shuffling reviews and corresponding labels within training and test sets
        self.data_train, self.labels_train = shuffle(self.data_train, self.labels_train)
        self.data_test, self.labels_test = shuffle(self.data_test, self.labels_test)

    def tokenize(self,text):
        """
        transforms raw text into a sequence of words
        input:
        text unprocessed
        output
        final list of words
        """
        #removing html tags
        text = BeautifulSoup(text, "html5lib").get_text()
        # convert to lower case
        text = text.lower()
        #remove non letters characters
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        #splitting
        words = text.split()
        #remove stopwords
        words = [w for w in words if w not in stopwords.words("english")]

        # Return the final list of words
        return words

    def encode_data(self):
        """
        Tokenize and encode data into a list of integers.
        input:
            training data
            training labels
        output:
            encoded trainnig features
            encoded training labels
        """

        #Tokenizing
        training_features = [self.tokenize(text) for text in self.data_train]

        # Initialize word2id and label2id dictionaries that will be used
        # to encode words and labels
        self.word2id = dict()
        self.label2id = dict()
        # maximum number of words in a sentence
        self.max_words = 0

        # Constructing a word2id dict
        for sentence in training_features:
            for word in sentence:
                # Adding words to the dictionnay if they are not referenced
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
        # When the length of the sentence is greater than max_words, updating max_words
        if len(sentence) > self.max_words:
            self.max_words = len(sentence)

        #Building label2id and id2label dictionaries - using a set to drop duplicates
        self.label2id = {label: i for i, label in enumerate(set(self.labels_train))}
        self.id2label = {key: value for value, key in self.label2id.items()}

        # Encode features and labels
        self.encoded_features = [[self.word2id[word] for word in sentence] for sentence in training_features]
        self.encoded_labels = [self.label2id[label] for label in self.labels_train]

        # Padding the encoded features so they are all the same length
        self.encoded_features = pad_sequences(self.encoded_features, self.max_words)

        # Converting the encoded labels to a matrix
        self.encoded_labels = to_categorical(self.encoded_labels, num_classes=len(self.label2id))


    def build_network(self):
        """
        Builds a network sequentially
        """
        #model stucture
        embedding_size = 32
        self.model = Sequential()
        self.model.add(Embedding(input_dim=len(self.word2id),
                                 output_dim = embedding_size,
                                 input_length=self.max_words))
        self.model.add(Bidirectional(LSTM(50, dropout=0.4, return_sequences=False),
                       merge_mode='concat'))
        self.model.add(Dense(len(self.label2id), activation='sigmoid'))


        #hyperparameters
        learning_rate = 0.003

        #compiling model
        self.model.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate),
                      metrics=['accuracy'])


    def train(self, batch_size = 256, epochs = 0):
        """
        Trains the network on the training dataset
        input:
        array of training features
        array of matching tags
        batch_size
        num of epochs
        return:
        trained classifier
        """

        # #Reserve some training data for validation
        # validation_features, validation_labels = \
        # self.encoded_features[:batch_size],self.encoded_labels[:batch_size]
        # train_features, train_labels = \
        # self.encoded_features[batch_size:],self.encoded_labels[batch_size:]

        #Train the model
        print("**Training in Progress**")
        self.model.fit(self.encoded_features, self.encoded_labels,
                       validation_split=0.2, verbose = 1,
                       batch_size=batch_size, epochs= epochs)



    def test(self):
        """
        Evaluates the accuracy of the network on the test set
        """

        self.data_test = [self.tokenize(text) for text in self.data_test]
        self.encoded_test_features = [[self.word2id.get("word", 0) for word in sentence] for sentence in self.data_test]
        self.encoded_test_labels = [self.label2id[label] for label in self.labels_test]
        # Padding the encoded features so they are all the same length
        self.encoded_test_features = pad_sequences(self.encoded_test_features, self.max_words)

        # Converting the encoded labels to a matrix
        self.encoded_test_labels = to_categorical(self.encoded_test_labels, num_classes=len(self.label2id))
        # returns loss and other metrics specified in model.compile()
        scores = self.model.evaluate(self.encoded_test_features, self.encoded_test_labels, verbose=0)
        # scores[1] should correspond to accuracy as we passed the metrics =['accuracy']
        print("Test accuracy:", scores[1])



    def load(self, file="Sentiment_Analysis.h5"):
        """
        Build, training, testing and saving the network
        to the cache
        input:
            optional - cache file as a .h5
        output:
            trained model
            optional - cache file as a .h5
        """

        #defining where to store cache files
        directory = os.path.join("cache", "sentiment_analysis")
        #Ensure the cache_directory exist
        os.makedirs(directory, exist_ok=True)

        #Checking for cache file and loading the model
        if os.path.exists(os.path.join(directory, file)):
            self.model = load_model(os.path.join(directory, file))
            print("**Model loaded from", file)

        #Building, training, testing and saving the model
        else:
            print("**Loading Data**")
            self.load_data()
            self.join_data()
            print("**Encoding Data**")
            self.encode_data()
            print("**Building Network**")
            self.build_network()
            print("**Training Network**")
            self.train()
            print("**Testing Network**")
            self.test()
            self.model.save(os.path.join(directory, file))
            print("**Model saved", os.path.join(directory, file), "**")















