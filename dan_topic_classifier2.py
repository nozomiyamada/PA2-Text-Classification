from topic_classifier import TopicClassifier
from pythainlp import word_tokenize
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib


class DANTopicClassifier2(TopicClassifier):

    def load_model(self, model_file_name:str):
        """
        load trained model, vocab_to_index(0-127812), index(0-11)_to_label
        """
        self.model, self.vocab_to_index, self.i_to_l = joblib.load(model_file_name)

    def classify(self, title_text:str) -> str:
        """
        tokenize title_text, get index of all words and pad with 0

        title_text = 'ไปกินอาหาร'
        > tokenized_title = ['ไป', 'กิน', 'อาหาร']
        > test_x = [15, 10, 27]
        > after padding [15, 10, 27, 0, 0,...]
        """
        tokenized_title = word_tokenize(title_text)
        test_x = [self.vocab_to_index[vocab] for vocab in set(tokenized_title) if vocab in self.vocab_to_index.keys()]
        test_x = np.array(pad_sequences([test_x], value=0, padding='post', maxlen=128))
        result = self.model.predict_proba(test_x)  # predict by model, result is 12 dim vector

        return self.i_to_l[np.argmax(result)]  # get the index of the most largest element, and convert to str
