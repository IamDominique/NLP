import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import nltk
from nltk import word_tokenize
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('punkt')
from nltk.corpus import brown



class POS:

    #declaring labeled corpus as a global variable
    labeled_corpus = nltk.corpus.brown.tagged_sents(tagset='universal')


    def features(self, sequence, index):
        """
        Extracts a list of features for each word in a sequence
        Input:
        sequence : the sequence as a list of strings
        index : the index of the word
        Return:
        feature dictionary
        """
        return {
            'word': sequence[index],
            'first': index == 0,
            'last': index == len(sequence) - 1,
            'capitalized': sequence[index][0].upper() == sequence[index][0],
            'all_caps': sequence[index].upper() == sequence[index],
            'all_lower': sequence[index].lower() == sequence[index],
            'prev_word': '' if index == 0 else sequence[index - 1],
            'next_word': '' if index == len(sequence) - 1 else sequence[index + 1],
            'with_hyphen': '-' in sequence[index],
            'numeric': sequence[index].isdigit(),
            }


    def remove_tag(self, labeled_corpus = labeled_corpus):
        """
        The brown corpus returns lists of word/tags tuple, this helper function
        removes the tags for the feature extraction stage

        input:
        list of tuples
        return:
        list of strings
        """
        return [w for w, t in labeled_corpus]

    def transform_to_datasets(self, labeled_corpus = labeled_corpus):
        """
        Creates a dataset from a corpus of tagged scentences
        Input:
        lists of word/tag tuples
        Return:
        list of features dictionaries (one per word)
        list of matching tags as strings
        """

        #initializing data
        features = []
        tags = []
        for sequence in labeled_corpus:
            for index in range(len(sequence)):
                features.append(self.features(self.remove_tag(sequence), index))
                tags.append(sequence[index][1])

        return features, tags


    def split(self, dataset = labeled_corpus, percentage = 0.2):
        """
        Helper function that splits the dataset into a training and a test set

        input:
        Dataset as a list
        size of the testing set in percent(float)
        return:
        training set as a list
        testing set as a list
        """

        #applying % to the source dataset
        split = int(percentage * len(dataset))
        #splitting
        self.training_set = dataset[:split]
        self.test_set = dataset[split:]


    def build_network(self):
        """
        Builds a network pipeline with a randomforest
        """
        self.clf = Pipeline([
            #using DictVectorizer to process the features dictionnaries
            ('vectorizer', DictVectorizer(sparse=False)),
            ('network', RandomForestClassifier(max_features=12))])




    def train(self):
        """
        Trains the network on a dataset
        input:
        list of features dictionnaries
        list of matching tags
        return:
        trained network
        """
        print("**Training in Progress**")
        self.clf.fit(self.train_features,self.train_labels)



    def test(self):
        """
        Evaluates the accuracy of the network on the test set
        """
        print("Test Accuracy: {:.3%}".format(self.clf.score(self.test_features[:10000],
                                         self.test_labels[10000])))



    def load(self):
        """
        Builds, trains and tests the network
        """
        #Processing data
        self.split()
        self.train_features, self.train_labels = self.transform_to_datasets(self.training_set)
        self.test_features, self.test_labels = self.transform_to_datasets(self.test_set)

        #Setting up the network
        self.build_network()
        #training
        self.train()
        #testing
        self.test()


    def tag(self,sentence):
        """
        function that tags a sentence
        input:
        sentence as a string
        return:
        zip of words/tags tuples
        """
        #tokenizing the sentence as a list of strings for processing
        sequence = word_tokenize(sentence)
        #Predicting the part of speech
        tags = self.clf.predict([self.features(sequence, index) for index in range(len(sequence))])

        return zip(sequence, tags)

    def display(self,sentence):
        """
        Prints the part of speech for a given sentence
        input:
        sentence as a string
        returns:
        n/a
        """
        res = list(self.tag(sentence))
        for value in res:
            print("{:<15}{:>3}".format(str(value[0]),str(value[1])))










