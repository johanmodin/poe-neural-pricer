# Encodes/labels the retrieved data to integer arrays
from sklearn import preprocessing
import configparser
import numpy as np
import os


class Encoder:
    def __init__(self, classes_path):
        self.classes_path = classes_path
        self.fullEncoder = self._read_encoder(classes_path)
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.encoders = {}
        self.param_descriptions = {}

    def encode(self, X_Y):
        self.fullEncoder = self._read_encoder(self.classes_path)
        if self.fullEncoder is None:
            print('Error: fullEncoder not loaded before trying to encode.')
            return None
        self._create_encoders()

        Xs = [item_value_tuple[0] for item_value_tuple in X_Y]
        encoded_X_Y = []
        encoded_array_length = len(list(self.fullEncoder.classes_))
        for item_nr in range(len(Xs)):
            # We encode our array with -1 to start with as that's our 'false'
            # and almost all of our elements will be false
            encoded = [-1 for n in range(encoded_array_length)]
            for key in Xs[item_nr]:
                if key in self.param_descriptions:
                    if self.param_descriptions[key] == 'array':
                        for value in Xs[item_nr][key]:
                            encoded[self.fullEncoder.transform([self._remove_digits(value)])[0]] = self._extract_digits(value)
                    elif self.param_descriptions[key] == 'string':
                        encoded[self.fullEncoder.transform([Xs[item_nr][key]])[0]] = 1
                    else:
                        print('Error: Could not find encoder for key %s' % key)
                else:
                    encoded[self.fullEncoder.transform([key])[0]] = self._get_key_value(Xs[item_nr][key])
            # append the (encoded X, value) tuple
            encoded_X_Y.append((np.array(encoded).astype('float32'), np.array(X_Y[item_nr][1]).astype('float32')))
        return np.array(encoded_X_Y)

    def fit(self, Xs):
        self.fullEncoder = self._read_encoder(self.classes_path)
        if self.encoders == {}:
            self._create_encoders()
        keys_to_fit = self._fit_encoders(Xs)

        classes = []
        if self.fullEncoder is not None:
            classes.extend(list(self.fullEncoder.classes_))

        for key in keys_to_fit:
            classes.extend(self.encoders[key].classes_)

        self.fullEncoder = preprocessing.LabelEncoder()
        self.fullEncoder.fit(classes)
        np.save(self.classes_path, self.fullEncoder.classes_)

        print('Fitted to %s mods.'
              % len(list(self.fullEncoder.classes_)))

    def _fit_encoders(self, Xs):
        # Someday this should be changed to not add duplicate mods.
        # It's not an issue for the labeller but it's not nice and uses
        # unnecessarily much RAM. This probably goes hand in hand with adding
        # functionality to enable adding more mods to a loaded fullEncoder.

        # Also, break this function up into subfunctions. It looks horrible.
        keys_to_fit = {}
        for item in Xs:
            for key in item:
                if key in self.param_descriptions:
                    if self.param_descriptions[key] == 'array':
                        for mod in item[key]:

                            # Remove all digits from the mods when we're encoding them
                            # to not get a lot of duplicate mods with different digit values
                            if key in keys_to_fit:
                                keys_to_fit[key].append(self._remove_digits(mod))
                            else:
                                keys_to_fit[key] = [self._remove_digits(mod)]
                    elif self.param_descriptions[key] == 'string':
                        # Encodes strings like typeline (base item type)
                        if key in keys_to_fit:
                            keys_to_fit[key].append(self._remove_digits(item[key]))
                        else:
                            keys_to_fit[key] = [self._remove_digits(item[key])]
                else:
                    if 'others' in keys_to_fit:
                        keys_to_fit['others'].append(key)
                    else:
                        keys_to_fit['others'] = [key]
        for key in keys_to_fit:
            if key in self.encoders:
                self.encoders[key].fit(keys_to_fit[key])
            else:
                self.encoders['others'].fit(key)
        return keys_to_fit

    def _get_key_value(self, string):
        clean_string = str(string).lower()
        if clean_string == 'true':
            return 1
        elif clean_string == 'false':
            return -1
        elif any(char.isdigit() for char in clean_string):
                return self._extract_digits(clean_string)
        else:
            print('Error: Could not get key value from string: %s' % string)
            return None

    def _remove_digits(self, string):
        return ''.join(c for c in string if not c.isdigit())

    def _extract_digits(self, string):
        values = [word for word in string.split(' ') if any(char.isdigit() for char in word)]
        clean_values = []
        for value in values:
            number = ''.join(c for c in value if c.isdigit() or c is '.')
            if len(number) > 0:
                clean_values.append(float(number))

        if len(clean_values) > 1:
            #print('debug: %s cleaned to %s' % (string, np.mean(clean_values)))
            return np.mean(clean_values)
        if len(clean_values) == 0:
            #print('debug: %s cleaned to 1' % (string))
            return 1
        #print('debug: %s cleaned to %s' % (string, clean_values))
        return np.mean(clean_values)

    def _read_encoder(self, path):
        if path is not None:
            if os.path.isfile(path):
                encoder = preprocessing.LabelEncoder()
                encoder.classes_ = np.load(path)
                return encoder
        else:
            print('Notice: No stored fullEncoder found.')
            return None

    def _create_encoders(self):
        self.encoders = {}
        self.param_descriptions = {}

        stringEncoder = preprocessing.LabelEncoder()
        stringKeys = self._get_config_value('stringKeys')
        for key in stringKeys:
            self.param_descriptions[key] = 'string'
            self.encoders[key] = stringEncoder

        arrayKeys = self._get_config_value('arrayKeys')
        array_encoders = []
        for key in arrayKeys:
            self.param_descriptions[key] = 'array'
            self.encoders[key] = preprocessing.LabelEncoder()
            array_encoders.append(self.encoders[key])

        othersEncoder = preprocessing.LabelEncoder()
        self.encoders['others'] = othersEncoder

    def _get_config_value(self, config_key):
        value = self.config['encoder'][config_key]

        if ',' in value:
            return value.split(',')
        return [value]
