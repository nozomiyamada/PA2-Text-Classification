from topic_classifier import TopicClassifier
from pythainlp import word_tokenize
import numpy as np
import joblib

class DANTopicClassifier(TopicClassifier):

    def load_model(self, model_file_name:str):
        self.model, self.wv, self.i_to_l = joblib.load(model_file_name)
        self.l_to_i = {v:k for k,v in self.i_to_l.items()}

    def classify(self, title_text:str) -> str:
        tokenized_title = word_tokenize(title_text)
        vecs = []
        for word in tokenized_title:
            if word in self.wv.vocab:
                vecs.append(self.wv[word])
        if vecs == []:
            mean = [0]*300
        else:
            mean = np.mean(np.array(vecs), axis=0)
        result = self.model.predict_proba(np.array([mean]))

        return self.i_to_l[np.argmax(result)]
