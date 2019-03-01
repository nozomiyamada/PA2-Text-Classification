import joblib
import numpy as np
from pythainlp import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
DV = DictVectorizer()

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

labels, titles = [], []
with open('title_classification.train', 'r', encoding='utf-8') as f:
    for line in f:
        label, title = line.split('\t', 1)
        labels.append(label)
        titles.append(word_tokenize(title.rstrip('\n')))
        
labels_dev, titles_dev = [], []
with open('title_classification.dev', 'r', encoding='utf-8') as f:
    for line in f:
        label, title = line.split('\t', 1)
        labels_dev.append(label)
        titles_dev.append(word_tokenize(title.rstrip('\n')))


### maximum entropy ###

def train_maxent():
    feature_matrix = []
    for title in titles:
        # if word is in sentence = 1, else 0
        feat_dic = {word: 1 for word in title}
        feature_matrix.append(feat_dic)

    sparse = DV.fit_transform(feature_matrix)  # sparse matrix for training data
    model = LogisticRegression(penalty='l2', C=1)
    model.fit(sparse, labels)
    joblib.dump((model, DV), 'maxent_model.bin')


### word embedding ###
    
wv = KeyedVectors.load_word2vec_format('skip.bin', binary=True)

# for DAN2
input_node = len(wv.vocab) + 1  # 127812 + 1
output_node = len(set(labels))  # 12

label_to_index = {label:i for i, label in enumerate(set(labels))}
index_to_label = {v:k for k, v in label_to_index.items()}

embedding_matrix = [[0]*300]
vocab_to_index = {0:'<pad>'}
for i, vocab in enumerate(wv.vocab.keys()):
    embedding_matrix.append(wv[vocab])
    vocab_to_index[vocab] = i+1


def prepare_dan1():

    train_vec = []
    for title in titles:
        vecs = [wv[word] for word in title if word in wv.vocab]
        if vecs == []:
            mean = np.zeros(300)
        else:
            mean = np.mean(np.array(vecs), axis=0)
        train_vec.append(mean)
    train_vec = np.array(train_vec).reshape((len(titles), 300))

    dev_vec = []
    for title in titles_dev:
        vecs = [wv[word] for word in title if word in wv.vocab]
        if vecs == []:
            mean = np.zeros(300)
        else:
            mean = np.mean(np.array(vecs), axis=0)
        dev_vec.append(mean)
    dev_vec = np.array(dev_vec).reshape((len(titles_dev), 300))
        
    return train_vec, dev_vec


def train_dan1(train_vec, dev_vec, epo=10, drop=0.2, act='relu'):

    index_list = [label_to_index[label] for label in labels]
    train_y = np.eye(output_node)[index_list]  # one-hot vectors
    index_list = [label_to_index[label] for label in labels_dev]
    val_y = np.eye(output_node)[index_list]  # one-hot vectors

    model = Sequential()

    # input 300 > hidden 100
    model.add(Dense(200, input_dim=300))
    model.add(Activation(act))
    model.add(Dropout(drop))
    
    # input 150 > hidden 50
    model.add(Dense(100))
    model.add(Activation(act))
    model.add(Dropout(drop))
    
    model.add(Dense(50))
    model.add(Activation(act))
    model.add(Dropout(drop))

    # hidden 50 > output 12
    model.add(Dense(output_node))
    model.add(Activation('softmax'))

    # optimizer = 'sgd', 'adam'
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_vec, train_y, verbose=1, validation_data=(dev_vec, val_y), epochs=epo)
    score = model.evaluate(train_vec, train_y, verbose=1)
    print('evaluate loss: {0[0]}\n evaluate acc: {0[1]}'.format(score))

    # plot accuracy
    plt.plot(history.history['acc'], marker='o', label='Training accuracy')
    plt.plot(history.history['val_acc'], marker='D', label='Validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()
    
    plt.figure()
    plt.plot(history.history['loss'], marker='o', label='Training loss')
    plt.plot(history.history['val_loss'], marker='D', label='Validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()
    
    joblib.dump((model, wv, index_to_label), 'dan_model.bin')


def prepare_dan2():
    train_x = []
    for i, title in enumerate(titles):
        indexes = [vocab_to_index[vocab] for vocab in set(title) if vocab in wv.vocab]
        train_x.append(indexes)
    train_x = pad_sequences(train_x, value=0, padding='post', maxlen=128)
    
    dev_x = []
    for i, title in enumerate(titles_dev):
        indexes = [vocab_to_index[vocab] for vocab in set(title) if vocab in wv.vocab]
        dev_x.append(indexes)
    dev_x = pad_sequences(dev_x, value=0, padding='post', maxlen=128)
    
    return np.array(train_x), np.array(dev_x)


def train_dan2(train_x, val_x, epo=3, drop=0.3, act = 'relu'):
    
    # make train_y = 12 dim one-hot vector   
    index_list = [label_to_index[label] for label in labels]
    train_y = np.eye(output_node)[index_list]  # make one-hot vectors
    
    index_list = [label_to_index[label] for label in labels_dev]
    val_y = np.eye(output_node)[index_list]  # make one-hot vectors
    
    model = Sequential()

    # input layer
    model.add(Embedding(input_node, 300, weights=[np.array(embedding_matrix)]))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(drop))
    
    # > hidden 200
    model.add(Dense(200))
    model.add(Activation(act))
    model.add(Dropout(drop))

    # > hidden 100
    model.add(Dense(100))
    model.add(Activation(act))
    model.add(Dropout(drop))
    
    # > hidden 50
    model.add(Dense(50))
    model.add(Activation(act))
    model.add(Dropout(drop))

    # > output 12
    model.add(Dense(output_node))
    model.add(Activation('softmax'))

    # optimizer = 'sgd', 'adam'
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y, verbose=1, validation_data=(val_x, val_y), epochs=epo)
    score = model.evaluate(train_x, train_y, verbose=1)
    print('evaluate loss: {0[0]}\n evaluate acc: {0[1]}'.format(score))

    # plot accuracy
    plt.plot(history.history['acc'], marker='o', label='Training accuracy')
    plt.plot(history.history['val_acc'], marker='D', label='Validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()
    
    plt.figure()
    plt.plot(history.history['loss'], marker='o', label='Training loss')
    plt.plot(history.history['val_loss'], marker='D', label='Validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.show()
    
    joblib.dump((model, vocab_to_index, index_to_label), 'dan2_model.bin')
