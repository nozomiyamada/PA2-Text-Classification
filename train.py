import csv
import joblib
import numpy as np
from pythainlp import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
DV = DictVectorizer()

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

with open('title_classification.train', 'r', encoding='utf-8') as f:
    lines = list(csv.reader(f, delimiter='\t'))

labels = [line[0] for line in lines]
titles = [word_tokenize(line[1]) for line in lines]  # tokenized 2D list


### maximum entropy ###

def train_maxent():
    feature_matrix = []
    for title in titles:
        # δ(w, s): if word is in sentence = 1, else 0
        feat_dic = {word: 1 for word in title}
        feature_matrix.append(feat_dic)

    sparse = DV.fit_transform(feature_matrix)  # sparse matrix for training data
    model = LogisticRegression(penalty='l2', C=1)
    model.fit(sparse, labels)
    joblib.dump((model, DV), 'maxent_model.bin')


### word embedding ###
    
wv = KeyedVectors.load_word2vec_format('skip.bin', binary=True)

def train_dan():

    title_vec = []
    for title in titles:
        sum_vec = 0
        number = 0
        for word in title:
            if word in wv.vocab:
                sum_vec += wv[word]
                number += 1
        if number == 0:
            mean = [0]*300
        else:
            mean = sum_vec/number
        title_vec.append(mean)
    return np.array(title_vec).reshape((len(labels), 300))

def train_dan1(title_vec, epo=5):
    
    output_node = len(set(labels))  # 12
    
    label_to_index = {label:i for i, label in enumerate(set(labels))}
    index_to_label = {v:k for k, v in label_to_index.items()}
    
    index_list = [label_to_index[label] for label in labels]
    train_y = np.eye(output_node)[index_list]  # one-hot vectors
    
    
    model = Sequential()

    # input 300 > hidden 100
    model.add(Dense(200, input_dim=300))
    model.add(Activation('sigmoid'))
    #model.add(Activation('relu'))
    
    # input 150 > hidden 50
    model.add(Dense(100))
    model.add(Activation('sigmoid'))
    
    model.add(Dense(50))
    model.add(Activation('sigmoid'))

    # hidden 50 > output
    model.add(Dense(output_node))
    model.add(Activation('softmax'))

    # optimizer = 'sgd', 'adam'
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(title_vec, train_y, verbose=1, validation_split=0.1, epochs=epo)
    score = model.evaluate(title_vec, train_y, verbose=1)
    print('evaluate loss: {0[0]}\n evaluate acc: {0[1]}'.format(score))

    # plot accuracy
    plt.plot(history.history['loss'], marker='o', label='Training loss')
    plt.plot(history.history['val_loss'], marker='D', label='Validation loss')
    #plt.plot(history.history['acc'], label='acc', ls='-', marker='o')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()
    
    joblib.dump((model, wv, index_to_label), 'dan_model.bin')
    
    
def train_dan2(title_vec, epo=1):
    
    input_node = len(wv.vocab)  # 127812
    output_node = len(set(labels))  # 12
    
    label_to_index = {label:i for i, label in enumerate(set(labels))}
    index_to_label = {v:k for k, v in label_to_index.items()}
    
    index_list = [label_to_index[label] for label in labels]
    train_y = np.eye(output_node)[index_list]  # one-hot vectors
    
    
    model = Sequential()
    
    embedding_matrix = []
    vocab_to_index = {}
    for i, vocab in enumerate(wv.vocab):
        embedding_matrix.append(wv[vocab])
        vocab_to_index[vocab] = i

    # input layer 300 dim vector
    model.add(Embedding(input_node, 300, weights=[np.array(embedding_matrix)]))
    #model.add(keras.layers.GlobalAveragePooling1D())

    # input 300 > hidden 100
    model.add(Dense(30))
    model.add(Activation('sigmoid'))
    #model.add(Activation('relu'))

    # hidden 100 > output
    model.add(Dense(output_node))
    model.add(Activation('softmax'))

    # optimizer = 'sgd', 'adam'
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(title_vec, train_y, verbose=1, validation_split=0.2, epochs=epo)
    score = model.evaluate(title_vec, train_y, verbose=1)
    print('evaluate loss: {0[0]}\n evaluate acc: {0[1]}'.format(score))

    # plot accuracy
    plt.plot(history.history['loss'], marker='o', label='Training loss')
    plt.plot(history.history['val_loss'], marker='D', label='Validation loss')
    #plt.plot(history.history['acc'], label='acc', ls='-', marker='o')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()
    
    joblib.dump((model, wv, index_to_label), 'dan2_model.bin')