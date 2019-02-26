from topic_classifier import TopicClassifier
import joblib
import csv
from pythainlp import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class MaxEntTopicClassifier(TopicClassifier):

    def load_model(self, model_file_name:str):
        self.model, self.DV = joblib.load(model_file_name)


    def classify(self, title_text:str) -> str:
        feat_dic = {word: 1 for word in word_tokenize(title_text)}
        result = self.model.predict(self.DV.transform([feat_dic]))
        return str(result[0])  # result = array(['คุณภาพชีวิต'], dtype='<U12')


    def accuracy(self):
        file = open('title_classification.dev', 'r', encoding='utf-8')
        lines = list(csv.reader(file, delimiter='\t'))
        titles_dev = [line[1] for line in lines]
        labels_dev = [line[0] for line in lines]
        tokenized_titles = [word_tokenize(x) for x in titles_dev]  # 2D list
        feature_matrix = []
        for title in tokenized_titles:
            feat_dic = {word: 1 for word in title}
            feature_matrix.append(feat_dic)
        result = self.model.predict(self.DV.transform(feature_matrix))
        accuracy = accuracy_score(labels_dev, result)
        print("Accuracy")
        print(accuracy)

