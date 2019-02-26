"""Topic Classifier base classes

You should not change anything in this file. 
This file essentially lists all of the methods you need to implement
for each subclass of TopicClassifier
"""

import joblib
import random

class TopicClassifier:
    """Base class for topic classifier

    The subclasses must implement the load_model and classify method
    load_model should load up all of the necessary parameters and store them internally
    classify should do whatever operation to classify a title text string to label string
    """

    def __init__(self, model_file_name:str=None):
        self.load_model(model_file_name)

    def load_model(self, model_file_name:str):
        pass

    def classify(self, title_text:str) -> str: 
        raise NotImplementedError

    def classify_list(self, title_text_list:list) -> list:
        """Classify a list of titles

        Some models are faster to classify a list of titles than for-loop over each one.
        You should override this method. 
        """
        return [self.classify(title) for title in title_text_list]



class RandomTopicClassifier(TopicClassifier):

    def load_model(self, model_file_name:str):
        """Load the parameters (model) from files

        The parameters for this classifier do not do anything useful.
        This is just an example
        """
        mapping, parameter = joblib.load(model_file_name)
        self.mapping = mapping
        random.seed(parameter)
        

    def classify(self, title_text:str) -> str:
        """Classify randomly

        This classifier returns a random label string
        """
        num_classes = len(self.mapping)
        classification = random.randint(0, num_classes - 1)
        return self.mapping[classification]

class MajorityTopicClassifier(TopicClassifier):

    def classify(self, title_text:str) -> str:
        return 'การเมือง'

