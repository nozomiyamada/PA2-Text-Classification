from topic_classifier import TopicClassifier
import joblib
from pythainlp import word_tokenize


class MaxEntTopicClassifier(TopicClassifier):

    def load_model(self, model_file_name:str):
        """
        load the trained model and instance of DictVectorizer (DV)
        DV contains the feature name (word to index)
        """
        self.model, self.DV = joblib.load(model_file_name)

    def classify(self, title_text:str) -> str:
        feat_dic = {word: 1 for word in word_tokenize(title_text)}  # make feature dictionary of one title
        result = self.model.predict(self.DV.transform([feat_dic]))  # predict with model
        return str(result[0])  # result = array(['คุณภาพชีวิต'], dtype='<U12')

