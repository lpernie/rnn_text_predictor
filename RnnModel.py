"""
=======================================================
Class implementing RNN model using Keras
=======================================================
"""
import os
import re
import heapq
import random
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
os.environ["KERAS_BACKEND"] = "tensorflow"


class RnnModel:
    """
    Class implementing RNN model using Keras_tools.
    """

    # You extend the original constructor of FraudDetectionMlp
    def __init__(self):
        self.dataset = None
        # All possible chars in your dataset.
        self.alphabet = None
        # This represents how you end a training example. E.g. I like the sun|I love ice-cream|
        self.end_sentence = "|"
        self.sequence_length = 80
        self.step_size = 4
        self.model = None
        self.optimizer = None
        self.history = {}
        self.x = None
        self.y = None
        self.x_val = None
        self.y_val = None
        self.use_validation = False

    def load_raw_text(self, fpath, text_col='text', cols_for_vetos=None):
        """
        Load a dataset. It could be a text file, or a json file.
        :param fpath: Path of the input file. Supported: json or txt file.
        :type fpath: ``str``
        :param text_col: col name containing the text in the json input file
        :type text_col: ``str``
        :param cols_for_vetos: filters to be applied to the input json
        :type cols_for_vetos: ``tuple(col_name, value_to_exclude)``
        """
        if '.json' in fpath:
            # Open the file
            print(f'Loading: {fpath} using {text_col} as column containing text.')
            f = open(fpath, "r")
            raw_data = json.load(f)
            # Perform a basic cleaning through a dataframe
            df = pd.DataFrame(raw_data)
            cols_for_vetos = [] if cols_for_vetos is not None else cols_for_vetos
            for filtering in cols_for_vetos:
                df = df.loc[df[filtering[0]] == filtering[1]]
            # Transform the dataframe into a string, where each sentence is separated by {self.end_sentence}
            raw_data = [sentence.replace(self.end_sentence, ' ').replace('\n', self.end_sentence)
                        for sentence in df[text_col].tolist()]
            raw_data = self.end_sentence.join(raw_data).lower()
        elif '.txt' in fpath:
            print(f'Loading: {fpath}.')
            raw_data = open(fpath, encoding="utf8").read().lower()
            raw_data = raw_data.replace(self.end_sentence, ' ').replace("\n", self.end_sentence)
        else:
            print(f'WARNING: returning dataset=None. File format is not understood, it should end with .json or .txt.')
            raw_data = None
        self.dataset = raw_data

    def veto_sentences(self, veto_string, where=None, veto_len=None):
        """
        Remove from the dataset sentences containing a substring, or belo a specific size.
        :param veto_string: substring to veto
        :type veto_string: ``str``
        :param where: where the substring is located (if None location doesn't matter)
        :type where: ``str``
        :param veto_len: min number of words in a sentence. None means no veto applied.
        :type veto_len: ``int``
        """
        veto_len = 999999 if veto_len is None else veto_len
        if where == 'start':
            tmp_dataset = [sentence for sentence in self.dataset.split(self.end_sentence)
                           if (not sentence.startswith(veto_string) and len(sentence) > veto_len)]
        else:
            tmp_dataset = [sentence for sentence in self.dataset.split(self.end_sentence)
                           if (veto_string not in sentence and len(sentence) > veto_len)]
        self.dataset = self.end_sentence.join(tmp_dataset)

    def remove_links(self):
        """
        Remove websites from the dataset.
        """
        self.dataset = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", self.dataset)

    def remove_emojis(self):
        """
        Remove non ascii characters from the dataset.
        """
        self.dataset = self.dataset.encode('ascii', 'ignore').decode('ascii')

    def remove_ellipsis(self):
        """
        Remove ellipsis from the dataset.
        """
        # Match .. followed by any repetition of . (so basically .. ... .... and so on will be converted into a space)
        self.dataset = re.sub('\.\.[\.]*', " ", self.dataset)
        self.dataset = re.sub('--[-]*', " ", self.dataset)

    def remove_parens(self):
        """
        Remove parenthesis from the dataset.
        """
        self.dataset = re.sub("[\(\[].*?[\)\]]", "", self.dataset)

    def remove_chars(self, chars_to_remove, subst_with=''):
        """
        Substitute specific chars from the dataset.
        :param chars_to_remove: substring to substitute.
        :type chars_to_remove: ``list of strings, or string``
        :param subst_with: string to substitute with.
        :type subst_with: ``string``
        """
        if isinstance(chars_to_remove, list):
            for elem in chars_to_remove:
                self.dataset = self.dataset.replace(elem, subst_with)
        else:
            self.dataset = self.dataset.replace(chars_to_remove, subst_with)

    def remove_spaces(self):
        """
        Remove multiple spaces from the dataset.
        """
        self.dataset = re.sub(' [ ]*', ' ', self.dataset)

    def concat(self, final_file,  *args):
        """
        Concat multiple clean txt files into a single dataset.
        :param final_file: final paths of the concatenated output
        :type final_file: ``strings``
        :param args: list of paths of the files to concatenate
        :type args: ``strings``
        """
        self.dataset = ''
        for item in args:
            if 'txt' in item:
                raw_data = open(item, encoding="utf8").read().lower()
                if raw_data[-1] == self.end_sentence:
                    self.dataset += raw_data
                else:
                    self.dataset += raw_data + self.end_sentence
            else:
                print(f'WARNING: skipping {item} since it is a different format than txt.')
        self.dataset = self.dataset.replace('\n', self.end_sentence)
        tc = open(final_file, "w", encoding="utf8")
        tc.write(self.dataset)
        tc.close()

    def load_clean_text(self, fpath):
        """
        Load a dataset. It could be a text file, or a json file.
        :param fpath: Path of the input file. Supported: txt file.
        :type fpath: ``str``
        """
        if '.txt' in fpath:
            print(f'Loading: {fpath}.')
            raw_data = open(fpath, encoding="utf8").read().lower()
        else:
            print(f'WARNING: returning dataset=NONE, as the file format is not understood. It should end with .txt.')
            raw_data = None
        self.dataset = raw_data
        self.alphabet = sorted(list(set(raw_data)))

    def prepare_feature_labels(self, validation_size=0.05, sequence_length=80, step_size=4):
        """
        From the loaded dataset, you create the X and Y matrices for training and evaluating.
        :param validation_size: size of the validation dataset
        :type validation_size: ``float``
        :param sequence_length: how much context you give to each prediction.
        :type sequence_length: ``int``
        :param step_size: how many chars are you moving to the right from one example to the next one
        :type step_size: ``int``
        """
        self.sequence_length = sequence_length
        self.step_size = step_size
        # sentectes: list of {Sequence_Length} chars. The next sentence is always {Step_Size} chars shifted
        sentences = []
        # next_chars: for each sentence this is the next letter you should predict
        next_chars = []

        for i in range(0, len(self.dataset) - self.sequence_length, self.step_size):
            sentences.append(self.dataset[i: i + self.sequence_length])
            next_chars.append(self.dataset[i + self.sequence_length])
        print(f' . Example of X and y: {sentences[0]} -> {next_chars[0]}')

        # X: matrix of size {n_sentences} x {SEQUENCE_LENGTH} x {unique_chars} (here it is initialized to all zeros)
        x = np.zeros((len(sentences), self.sequence_length, len(self.alphabet)), dtype=np.bool)
        # Y: a matrix of size {n_sentences} x {unique_chars} (here it is initialized to all zeros)
        y = np.zeros((len(sentences), len(self.alphabet)), dtype=np.bool)

        # Each letter is a vector of size {unique_chars} with all zeros and a single one.
        # It means in the first sequence of X the second letter is 'd' and d -> 3, then X[1][2] = [0,0,0,1,0,0,..0]
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, self._char_indices[char]] = 1
            y[i, self._char_indices[next_chars[i]]] = 1

        print(f'X has shape {x.shape}, and y has shape {y.shape}')
        if 0 < validation_size < 1:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=validation_size)
            self.x = x_train
            self.y = y_train
            self.x_val = x_test
            self.y_val = y_test
            self.use_validation = True
        else:
            self.x = x
            self.y = y
            self.x_val = None
            self.y_val = None

    def build_model(self, lstm_out_size=None, next_layers=None):
        """
        Build a model.
        :param lstm_out_size: output size of the first LSTM layer
        :type lstm_out_size: ``int``
        :param next_layers: list containing the output space of each hidden layer
        :type next_layers: ``list of int``
        """
        self.model = Sequential()
        # The first part is a LSTM model. You feed a vector {SEQUENCE_LENGTH}x{unique_chars} and you get an abstract
        # output vector of size {unique_chars x 5}.
        # The output size is arbitrary since is a abstarct representation of the prediction in a latent space
        self.model.add(LSTM(lstm_out_size, input_shape=(self.sequence_length, len(self.alphabet))))
        next_layers = [] if next_layers is None else next_layers
        for size_layer in next_layers:
            self.model.add(BatchNormalization())
            self.model.add(Activation('selu'))
            # The next layers will decode the latent space of the predictions and will convert it into a letter
            # The outer space of this layer is {uniqe_chars x 2}, that is still arbitrary.
            self.model.add(Dense(size_layer))
        # The final layer has to be a softmax on a vector of size {unique_chars}.
        self.model.add(BatchNormalization())
        self.model.add(Activation('selu'))
        self.model.add(Dense(len(self.alphabet)))
        self.model.add(Activation('softmax'))

        self.optimizer = RMSprop(lr=0.001)

    def compile_and_fit(self, epochs, callbacks):
        """
        Compile and fit the model.
        :param epochs: number of epochs
        :type epochs: ``int``
        :param callbacks: list of callbacks you want to use
        :type callbacks: ``list``
        """
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer,
                           metrics=['accuracy'])
        if self.use_validation:
            self.history = self.model.fit(self.x, self.y, validation_data=(self.x_val, self.y_val),
                                          batch_size=124, epochs=epochs,
                                          shuffle=True, callbacks=callbacks)
        else:
            self.history = self.model.fit(self.x, self.y,
                                          batch_size=124, epochs=epochs,
                                          shuffle=True, callbacks=callbacks)

    def load_model(self, model_to_load, history=None):
        """
        Compile and fit the model.
        :param model_to_load: path to the model to load
        :type model_to_load: ``str``
        :param history: path to the history to load
        :type history: ``str``
        """
        self.model = load_model(model_to_load)
        if history is not None:
            if 'log' in history:
                self.history = pd.read_csv(history, sep=',', engine='python')
            elif 'json' in history:
                f = open(history, "r")
                self.history = json.load(f)
            else:
                print(f'WARNING: I am not loading the history, since I do not understand the format')

    @staticmethod
    def sample(preds, multinomial_thresh=1.3):
        preds = np.asarray(preds).astype('float64')  # convert to array
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        max_prob = max(preds)
        second_max = max(np.delete(preds, max_prob))
        extract_polinomial = False if (max_prob / second_max) > multinomial_thresh else True
        if extract_polinomial:
            return random.choices(range(0, len(preds)), preds, k=1)[0]
        else:
            return heapq.nlargest(1, range(len(preds)), preds.take)[0]

    def prepare_input(self, text):
        x = np.zeros((1, self.sequence_length, len(self.alphabet)))
        for t, char in enumerate(text):
            x[0, t, self._char_indices[char]] = 1.
        return x

    def predict_completions(self, text, next_is, multinomial_thresh=1.3):
        word = ''
        need_to_end = False
        while not need_to_end:
            x = self.prepare_input(text)
            preds = self.model.predict(x, verbose=0)[0]  # Return [[w1,w2,..,w_n_unique_chars]]
            next_indices = self.sample(preds, multinomial_thresh)  # Return the next index
            letter = self._lett_indices[next_indices]
            text = text[1:] + letter
            word += letter
            if next_is == 'world' and letter == ' ':
                need_to_end = True
            if next_is == 'letter':
                need_to_end = True
        return word

    def gen_sentence(self, text, next_words=2, next_letters=None, multinomial_thresh=1.3):
        """
        Predict the next words based on an initial text.
        :param text: Initial text that offers context
        :type text: ``str``
        :param next_words: how many worlds you want to predict. None if you used next_letters.
        :type next_words: ``int``
        :param next_letters: how many letters you want to predict. None if you used next_words.
        :type next_letters: ``int``
        :param multinomial_thresh for sampling the next letter using a multinomial distribution
        :type multinomial_thresh: ``float``
        """
        how_many_pred = next_letters
        if next_words is None and next_letters is not None:
            next_is = 'letter'
        elif next_words is not None and next_letters is None:
            next_is = 'world'
            how_many_pred = next_words
        else:
            next_is = 'letter'
            print(f'WARNING you are either predicting N words, either N letters. Predicting {next_letters} letters,.')
        text_og = text
        text = text.lower()
        while len(text) < self.sequence_length:
            text = ' ' + text
        text = text[-self.sequence_length:]
        for i in range(how_many_pred):
            text = text[-self.sequence_length:]
            pred = self.predict_completions(text, next_is, multinomial_thresh)
            text = text + pred
            text_og = text_og + pred
            pass
        return text_og

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    @property
    def alphabet(self):
        return self._alphabet

    @alphabet.setter
    def alphabet(self, values):
        self._alphabet = values
        if values is not None:
            self._char_indices = dict((c, i) for i, c in enumerate(values))
            self._lett_indices = dict((i, c) for i, c in enumerate(values))

    @property
    def sequence_length(self):
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self._sequence_length = value

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, value):
        self._step_size = value

    @property
    def end_sentence(self):
        return self._end_sentence

    @end_sentence.setter
    def end_sentence(self, value):
        self._end_sentence = value

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def x_val(self):
        return self._x_val

    @x_val.setter
    def x_val(self, value):
        self._x_val = value

    @property
    def y_val(self):
        return self._y_val

    @y_val.setter
    def y_val(self, value):
        self._y_val = value

    @property
    def use_validation(self):
        return self._use_validation

    @use_validation.setter
    def use_validation(self, value):
        self._use_validation = value

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history = value
