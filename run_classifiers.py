"""Run all of the classifiers on training and dev data

To make sure that their accuracies look OK.
You must not change anything in this file.
"""
from sklearn.metrics import accuracy_score, classification_report

from topic_classifier import TopicClassifier, MajorityTopicClassifier, RandomTopicClassifier
from maxent_topic_classifier import MaxEntTopicClassifier
from dan_topic_classifier import DANTopicClassifier
from dan_topic_classifier2 import DANTopicClassifier2

def read_data(data_file_name:str):
    labels = []
    titles = []
    with open(data_file_name) as f:
        for line in f:
            label, title = line.split('\t', 1)
            labels.append(label)
            titles.append(title)
    return (labels, titles)       

def evaluate_classifier(classifier:TopicClassifier, data_file_name:str):
    true_labels, titles = read_data(data_file_name)
    prediction = classifier.classify_list(titles)
    #print (classification_report(true_labels, prediction))
    print (accuracy_score(true_labels, prediction))

if __name__ == '__main__':
    """
    rtc = RandomTopicClassifier('random_model.bin')
    print(rtc.classify('นายกลุยภาคใต้'))
    print('Random classification train and test accuracies')
    evaluate_classifier(rtc, 'title_classification.train')
    evaluate_classifier(rtc, 'title_classification.dev')

    mjc = MajorityTopicClassifier()
    print('Majority classification train and test accuracies')
    evaluate_classifier(mjc, 'title_classification.train')
    evaluate_classifier(mjc, 'title_classification.dev')
    
    maxent = MaxEntTopicClassifier('maxent_model.bin')
    print('MaxEnt model train and test accuracies')
    evaluate_classifier(maxent, 'title_classification.train')
    evaluate_classifier(maxent, 'title_classification.dev')
    """
    dan1 = DANTopicClassifier('dan_model.bin')
    print('DAN model train and test accuracies')
    evaluate_classifier(dan1, 'title_classification.train')
    evaluate_classifier(dan1, 'title_classification.dev')

    dan2 = DANTopicClassifier2('dan2_model.bin')
    print ('DAN model2 train and test accuracies')
    evaluate_classifier(dan2, 'title_classification.train')
    evaluate_classifier(dan2, 'title_classification.dev')
