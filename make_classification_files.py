"""Run all of the classifiers on dev and test data

You need to submit these files with your code.
"""
from topic_classifier import TopicClassifier
from dan_topic_classifier import DANTopicClassifier
from dan_topic_classifier2 import DANTopicClassifier2
from run_classifiers import read_data


def classify_data_file(classifier:TopicClassifier, 
        data_file_name:str,
        output_file_name:str):
    _, titles = read_data(data_file_name)
    prediction_list = classifier.classify_list(titles)
    with open(output_file_name, 'w') as out:
        for p in prediction_list:
            out.write('{}\n'.format(p))

    
if __name__ == '__main__':

    dan1 = DANTopicClassifier('dan_model.bin')
    classify_data_file(dan1, 'title_classification.dev', 'dan_model.dev.classification')
    classify_data_file(dan1, 'nolabel_title_classification.test', 'dan_model.test.classification')

    dan2 = DANTopicClassifier2('dan2_model.bin')
    classify_data_file(dan2, 'title_classification.dev', 'dan2_model.dev.classification')
    classify_data_file(dan2, 'nolabel_title_classification.test', 'dan2_model.test.classification')