# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing phonetic transcription
Padding is handled by using a repeated sentinel character (space)

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.

'''

from __future__ import print_function
from keras.utils.visualize_util import plot
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Reshape, Convolution1D, MaxPooling1D, Dropout, Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding, Reshape
import numpy as np
from six.moves import range
import subprocess
import matplotlib
from time import gmtime, strftime
matplotlib.use("pdf")
import matplotlib.pyplot as plt

def levenshtein(s, t):
    """
        levenshtein(s, t) -> ldist
        ldist is the Levenshtein distance between the strings
        s and t.
        For all i and j, dist[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    rows = len(s)+1
    cols = len(t)+1
    dist = np.zeros((rows, cols), dtype=np.int)
    # source prefixes can be transformed into empty strings
    # by deletions:
    for i in range(1, rows):
        dist[i,0] = i
    # target prefixes can be created from an empty source string
    # by inserting the characters
    for i in range(1, cols):
        dist[0,i] = i

    row = rows - 1
    col = cols - 1
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row, col] = min(dist[row-1, col] + 1,      # deletion
                                 dist[row, col-1] + 1,      # insertion
                                 dist[row-1, col-1] + cost) # substitution

    return dist[row, col]

def vectorization(words, trans):
    X = np.zeros((len(words), words_maxlen), dtype='int32')
    y = np.zeros((len(trans), trans_maxlen), dtype='int32')
    for i, word in enumerate(words):
        X[i,:] = ctable.encode(word)
    for i, tran in enumerate(trans):
        y[i,:] = ptable.encode(tran)
    if INVERT:
        X = X[:,::-1]
    return X, y

def save(refs, preds, filename):
    with open(filename, 'wt') as res:
        for ref, pred in zip(refs, preds):
            correct = ptable.decode(ref, ch=' ').strip()
            guess = ptable.decode(pred, calc_argmax=False, ch=' ').strip()
            print(correct, '|', guess, file=res)

class Dictionary(dict):
    def __init__(self, filename):
        with open(filename, 'rt') as f:
            for line in f:
                word, phones = line.split('\t')[:2]
                self.update({word: phones})
        super(Dictionary, self).__init__()

class CharacterTable(object):
    def __init__(self, chars, maxlen):
        self.chars = [''] + sorted(set(chars))
        self.indices = dict((c, i) for i, c in enumerate(self.chars))
        self.maxlen = maxlen
        self.size = len(self.chars)

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros(maxlen, dtype='int32')
        for i, c in enumerate(C):
            X[i] = self.indices[c]
        return X

    def decode(self, X, calc_argmax=False, ch=''):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ch.join(self.chars[x] for x in X)

def calcPerformance(iteration):
    # dummy quotes
    dqote = '"'
    sqote = "'"

    namefile = "rnn_" + str(iteration) + ".pred"
    cmdTasas = "./tasas " + namefile + " -F -f " + sqote + "|" + sqote + " -s " + dqote + " " + dqote + " -pra"

    output = subprocess.Popen([cmdTasas], shell=True, stdout=subprocess.PIPE).communicate()[0]
    subprocess.Popen(["rm " + namefile], shell=True, stdout=subprocess.PIPE).communicate()[0]
    output = output.splitlines()[1]
    output = output.split()

    goles = output[1]
    goles = goles[:-1]
    subs = output[3]
    subs = subs[:-1]
    ins = output[5]
    ins = ins[:-1]
    borr = output[7]
    borr = borr[:-1]

    Scores = np.zeros(4)
    Scores[0] = float(goles)
    Scores[1] = float(subs)
    Scores[2] = float(ins)
    Scores[3] = float(borr)

    return Scores

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# ------------------------- VARIABLES DEFINITION AND INITIALIZATION ------------------------- #
print("\n\n")
print("------------------------- Configuration -------------------------")

# Configuration
isAligned = raw_input("NOT Aligned (NA), Aligned (A): ")
isConvolutional = raw_input("NOT Conv (NC), Conv (C): ")
isBid = raw_input("NOT Bidirectional (NB), Bidirectional (B): ")
nNeurons = int(raw_input("Number of neurons: "))

# Directory to store results
currentdata = strftime("%d_%H-%M-%S", gmtime())
dirname = isAligned + "_" + isConvolutional + "_" + isBid + "_N-" + str(nNeurons)
dirname = dirname + "__" + currentdata
cmdmkdir = "mkdir " + dirname
subprocess.Popen([cmdmkdir], shell=True, stdout=subprocess.PIPE).communicate()[0]

# Parameters for the model and dataset
if isAligned == 'A':
    TRAIN='wcmudict.train.aligned'
    TEST='wcmudict.test.aligned'
else:
    TRAIN='wcmudict.train.dict'
    TEST='wcmudict.test.dict'

# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
VSIZE = 7   # Embedded vector size
HIDDEN_SIZE = nNeurons   # Size of hidden layer after first RNN
BATCH_SIZE = 128
LAYERS = 1  # Layers of the second RNN
INVERT = True   # Invert the word

# Split the database (in % of database split, all database = 1)
DB_SPLIT = float(raw_input("Portion of the db that you want to use (all database = 1): "))
N_ITER = int(raw_input("Number of epochs: "))  # Number of epochs

train = Dictionary(TRAIN)
test = Dictionary(TEST)

words = map(str.split, train.keys())    # [['u', 's', 'e', 'd'], ['m', 'o', 'u', 'n', 't'], ...]
trans = map(str.split, train.values())
words_test = map(str.split, test.keys())
trans_test = map(str.split, test.values())

words_maxlen = max(map(len, words))
trans_maxlen = max(map(len, trans))
words_t_maxlen = max(map(len, words_test))
trans_t_maxlen = max(map(len, trans_test))

words_maxlen = max([words_maxlen, words_t_maxlen])
trans_maxlen = max([trans_maxlen, trans_t_maxlen])

print('Maxlen: ',format(trans_maxlen))

# These are all the used characters ("set" does not accept duplicates)
chars = sorted(set([char for word in words for char in word]))

# These are all the used phones (in the aligned db, not all the phonemes are present in the train db)
phones = set()
for xx, tran in enumerate(trans):
    for jj, c in enumerate(tran):
        phones.add(c)

print("Size train pho: " + str(len(phones)))
for xx, tran_t in enumerate(trans_test):
    for jj, c in enumerate(tran_t):
        phones.add(c)

print("Size full pho: " + str(len(phones)))
phones = sorted(phones)


ctable = CharacterTable(chars, words_maxlen)    # ctable = {'': 0, "'": 1, '-': 2, '.' : 3}
ptable = CharacterTable(phones, trans_maxlen)   # El mateix pero amb fonemes


print("\n\n")
print('Total training words:', len(words))

print('Vectorization...')

# X_train[1,:] vector on cada entrada es l'index de la lletra (pero esta girat)
# Y_train[1,:] vector on cada entrada es l'index del phonema
X_train, y_train = vectorization(words, trans)

# Use less data
new_dbl = int(len(X_train[:,1])*DB_SPLIT)
X_train = X_train[1:new_dbl, :]
y_train = y_train[1:new_dbl, :]

X_val, y_val = vectorization(words_test, trans_test)

# Shuffle (X_train, y_train) in unison
indices = np.arange(int (len(y_train)*DB_SPLIT))
np.random.seed (0)
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

print(X_train.shape)
print(y_train.shape)

# ------------------------------- MODEL DEFINITION -------------------------------- #
print('Build model...')
model = Sequential()

# Embedding of the characters to an VSIZE vector
layerEmbedding = Embedding(ctable.size, VSIZE, input_dtype='int32')
# If we want to set previous weights
# layerEmbedding.set_weights(np.load('weigths_Embedding.npy'))
model.add(layerEmbedding)



#PONER CONVOLOCIONAL 1D PARA REDUCIR EL TAMAÑO
#NOS DARA UNA MATRIZ MAS PEQUEÑA PARA HACER DIMENSION ENTRE PARAMETRES

#POR SI VA LENTO
#model.add(Dropout(0.25))


# Convolution
filter_length = 3
nb_filter = 15
pool_length = 4

# model.add(Convolution1D(nb_filter=nb_filter,
#                        filter_length=filter_length,
#                        border_mode='valid',
#                        activation='relu',
#                        subsample_length=1))
#model.add(MaxPooling1D(pool_length=pool_length))


# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
layerRNN1 = RNN(HIDDEN_SIZE)
model.add(layerRNN1)
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(trans_maxlen))
# The decoder RNN could be multiple layers stacked or a single layer
# layerRNN2 = {}
# for i in range(LAYERS):
layerRNN2 = RNN(HIDDEN_SIZE, return_sequences=True)
# POSAR AQUI UNA BIDIRECCIONAL model.add(Bidirecional(RNN...
model.add(layerRNN2)

# For each of step of the output sequence, decide which phone should be chosen
layerDense = TimeDistributed(Dense(ptable.size))
model.add(layerDense)

model.add(Activation('softmax'))
model.summary()
plot(model, show_shapes=True, to_file='pho_rnn.png', show_layer_names=False)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


def saveResults (measurements, N_ITER):
    print("Saving results...")
    # Save weights
    np.save([dirname + "/" + "weigths_Embedding.npy"], layerEmbedding.get_weights())
    np.save(dirname + "/" + "weigths_RNN1.npy", layerRNN1.get_weights())

    # Create and save plots

    currentdata = strftime("%H-%M-%S", gmtime())
    # Save results
    strmatrix = dirname + "/" + "res-" + currentdata + ".txt"
    np.savetxt(strmatrix, measurements)

    # Plot results
    plt.plot(range(1, N_ITER + 1), measurements[:, 0], label="Goals")  # Goals
    plt.plot(range(1, N_ITER + 1), measurements[:, 1], label="Subs")  # Substitutions
    plt.plot(range(1, N_ITER + 1), measurements[:, 2], label="Ins")  # Insertions
    plt.plot(range(1, N_ITER + 1), measurements[:, 3], label="Borr")  # Borrades
    plt.axis([1, N_ITER, 0, 100])
    plt.ylabel("%")
    plt.legend(bbox_to_anchor=(0.25, 1))
    plt.xlabel("Number of epochs")

    strplot = dirname + "/" + "plot-" + currentdata + ".png"
    plt.savefig(strplot)
    plt.clf()


# ----------------------------- TRAINING AND TESTING ------------------------------------- #

measurements = np.zeros((N_ITER, 4))

# Train the model each generation and show predictions against the validation dataset
for iteration in range(0,N_ITER):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train[..., np.newaxis], batch_size=BATCH_SIZE, nb_epoch=1,
              validation_data=(X_val, y_val[..., np.newaxis])) # add a new dim to y_train and y_val to match output
    preds = model.predict_classes(X_val, verbose=0)

    print('Saving results...')
    save(y_val, preds, 'rnn_{}.pred'.format(iteration))
    # Per a evitar que els weights s'actualitzin hem de freeze la layer, si creiem que ja son prou bons
    layerEmbedding.__setattr__("trainable", False)

    # Compute performance
    measurements[iteration, :] = calcPerformance(iteration)
    print(measurements[iteration, :])

    if iteration % 5 == 0:
        # Save just in case forced stop
        saveResults(measurements, N_ITER)

    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[ind], y_val[ind]
        pred = preds[ind]
        q = ctable.decode(rowX)
        correct = ptable.decode(rowy, ch=' ').strip()
        guess = ptable.decode(pred, ch=' ').strip()
        print('W:', q[::-1] if INVERT else q)
        print('T:', correct)
        print(colors.ok + '☑' + colors.close if correct == guess else colors.fail + '☒' + colors.close,
              guess, '(' + str(levenshtein(correct.split(), guess.split())) + ')')
        print('---')

# --------------------------------- SAVE FINAL RESULTS ----------------------------------- #
saveResults(measurements, N_ITER)
