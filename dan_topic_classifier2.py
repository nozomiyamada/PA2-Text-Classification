from topic_classifier import TopicClassifier

class DANTopicClassifier2(TopicClassifier):

    def load_model(self, model_file_name:str):
        pass

    def classify(self, title_text:str) -> str: 
        return 'การเมือง'