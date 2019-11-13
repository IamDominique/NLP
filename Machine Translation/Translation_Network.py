import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy


class Network:



    def load_data(self, path):
        """
        Load dataset
        """
        input_file = os.path.join(path)
        with open(input_file, "r") as f:
            data = f.read()

        return data.split('\n')



    def get_metrics(self,source_sentences, target_sentences):

        source_words_counter = collections.Counter([word for sentence in source_sentences for word in sentence.split()])
        target_words_counter = collections.Counter([word for sentence in target_sentences for word in sentence.split()])

        print(f"{len([word for sentence in source_sentences for word in sentence.split()])} source words.")
        print(f"{len(source_words_counter)} unique source words.")
        print("10 Most common words in the source dataset:")
        print('"' + '" "'.join(list(zip(*source.most_common(10)))[0]) + '"')
        print()
        print(f"{len([word for sentence in french_sentences for word in sentence.split()])} target words.")
        print(f"{len(french_words_counter)} unique target words.")
        print('10 Most common words in the target dataset:')
        print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')



    def tokenize(self,x):
        """
        Tokenize x
        :param x: List of sentences/strings to be tokenized
        :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
        """
        #initializing keras'tokenizer
        tokenizer = Tokenizer()
        #creating a vocabulary index based on word frequency
        tokenizer.fit_on_texts(x)
        #tokenizing sequence, i.e. transforms each sequence to a sequence of integers
        sequence = tokenizer.texts_to_sequences(x)

        return sequence, tokenizer


    def pad(self,x, length=None):
        """
        Pad x
        :param x: List of sequences.
        :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
        :return: Padded numpy array of sequences
        """
        while length is None:
            length = length = max([len(sequence) for sequence in x])
        #padding sequences
        res = pad_sequences(x, maxlen=length, padding='post')

        return res


    def preprocess(self, x, y):
        """
        Preprocess x and y
        :param x: Feature List of sentences
        :param y: Label List of sentences
        :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
        """
        self.preprocess_x, self.x_tk = self.tokenize(x)
        self.preprocess_y, self.y_tk = self.tokenize(y)

        self.preprocess_x = self.pad(self.preprocess_x)
        self.preprocess_y = self.pad(self.preprocess_y)

        # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
        self.preprocess_y = self.preprocess_y.reshape(*self.preprocess_y.shape, 1)



    def split_data(self):
        """
        Splits the data intro a training and a test set
        :param x: preprocessed source data
        :param y: preprocessed target data
        :return: Tuple of (x_train, y_train, x_test, y_test)
        """
        self.x_train, self.x_test, self.y_train, self.y_test = \
        train_test_split(self.preprocess_x, self.preprocess_y, test_size = 0.2, random_state = 0)



    def logits_to_text(self,logits, tokenizer):
        """
        Turn logits from a neural network into text using the tokenizer
        :param logits: Logits from a neural network
        :param tokenizer: Keras Tokenizer fit on the labels
        :return: String that represents the text of the logits
        """
        index_to_words = {id: word for word, id in tokenizer.word_index.items()}
        index_to_words[0] = '<PAD>'
        index_to_words[-1] = '<UNK>'

        return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


    def model(self,input_shape, output_sequence_length, source_vocab_size, target_vocab_size):
        """
        Build a model sequentially
        :param input_shape: Tuple of input shape
        :param output_sequence_length: Length of output sequence
        :param english_vocab_size: Number of unique English words in the dataset
        :param french_vocab_size: Number of unique French words in the dataset
        :return: Keras model built, but not trained
        """

        self.model = Sequential()

        ###Encoder network - takes the input sequence and maps it to an encoded representation
        #Embedding layer
        self.model.add(Embedding(input_dim=source_vocab_size, output_dim=128,
                            input_length=input_shape[1:][0]))
        self.model.add(Bidirectional(GRU(518, return_sequences=False),
                                input_shape=input_shape[1:]))
        self.model.add(Dense(target_vocab_size, activation="relu"))

        #repeating the vector for each time steps so it can be fed to the decoder
        self.model.add(RepeatVector(output_sequence_length))

        ###Decoder network - use the encoded representation to generate an output sequence
        self.model.add(Bidirectional(GRU(source_vocab_size, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(target_vocab_size, activation="softmax")))

        #hyper parameters
        learning_rate = 0.003

        #compiling model
        self.model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=Adam(learning_rate),
                      metrics=['accuracy'])

        return self.model


    def train(self, num_epoch = 10):
        """
        Train the model
        :param x: preprocessed source data
        :param y: preprocessed target data
        :param x_tk: Source tokenizer
        :param y_tk: Target tokenizer
        :param epoch: Number of epoch
        """
        self.model = self.model(self.x_train.shape,
                                self.y_train.shape[1],
                                len(self.x_tk.word_index) + 1,
                                len(self.y_tk.word_index) + 1)

        self.model.fit(self.x_train, self.y_train, batch_size=1024, epochs= num_epoch, validation_split=0.2)


    def test(self):
        """
        Evaluates the accuracy of the model on the test set
        :param x: Preprocessed source data
        :param y: Preprocessed target data
        """

        # returns loss and other metrics specified in model.compile()
        scores = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        # scores[1] should correspond to accuracy as we passed the metrics =['accuracy']
        print("Test accuracy:", scores[1])


    def predict(self, text):
        """
        """
        #preprocessing
        sentence = [self.x_tk.word_index.get(word, -1) for word in text.split()]
        sentence = pad_sequences([sentence], maxlen=self.preprocess_x.shape[-1], padding='post')
        #predicting
        predictions = self.model.predict(sentence, len(sentence))
        #building output
        output = self.logits_to_text(predictions[:1][0], self.y_tk)
        print(output)
