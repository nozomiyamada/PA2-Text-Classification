from topic_classifier import TopicClassifier
from pythainlp import word_tokenize
import numpy as np
import joblib


class DANTopicClassifier(TopicClassifier):

    def load_model(self, model_file_name:str):
        """
        load trained model, word embedding (gensim), index_to_label
        """
        self.model, self.wv, self.i_to_l = joblib.load(model_file_name)

    def classify(self, title_text:str) -> str:
        """
        tokenize title_text, get vectors of all words and append to the list
        if there are only UNK, returns zero vector

        title_text = 'ไปกินอาหาร'
        > tokenized_title = ['ไป', 'กิน', 'อาหาร']
        > vecs = [[11,2,5...], [3,-1,4...], [6,2,9...]]
        > np.mean(vecs, axis=0) = [2,1,5,...]
        """
        tokenized_title = word_tokenize(title_text)
        vecs = []  # make the list of each word vector
        for word in tokenized_title:
            if word in self.wv.vocab:  # append vector iff the word is in vocab
                vecs.append(self.wv[word])
        if vecs == []:  # if there is no vector in list, return [0,0,...0]
            mean = np.zeros((300))
        else:
            mean = np.mean(np.array(vecs), axis=0)  # calculate mean along column
        result = self.model.predict_proba(np.array([mean]))  # predict by model, result is 12 dim vector

        return self.i_to_l[np.argmax(result)]  # get the index of the most largest element, and convert to str
