# Filters the response content for what we're interested in

import configparser
import json
import copy
KEYS_WITH_DICT_VALUES = ['properties', 'requirements', 'sockets']

class Filter:
    def __init__(self, currency_converter, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.filter = self._setup_filter()
        self.required_keys = self._get_required_keys()
        self.retained_keys = self._get_retained_keys()
        self.currency_converter = currency_converter


    def filter_items(self, stashes):
        retained_items = []
        for stash in stashes:
            if 'items' in stash and stash['items'] is not []:
                for item in stash['items']:
                    if self._check_item_eligibility(item):
                        retained_items.append(item)
        retained_items = self._remove_unwanted_keys(retained_items)
        return retained_items

    def _remove_unwanted_keys(self, items):
        data = []
        for item in items:
            item_value = self.currency_converter.convert(item['note'])
            if item_value is -1:
                continue
            new_item = copy.deepcopy(item)
            for key in item:
                if key not in self.retained_keys:
                    del new_item[key]
                elif key in KEYS_WITH_DICT_VALUES:
                    for subkey in item[key]:
                        if subkey not in self.retained_keys[key]:
                            del new_item[key][subkey]
            data.append((new_item, item_value))
        return data

    def _check_item_eligibility(self, item):
        for key in self.required_keys:
            if key not in item:
                return False
        for key in self.filter:
            if key not in item:
                return False
            if str(item[key]).lower() not in self.filter[key]:
                return False
        return True

    def _setup_filter(self):
        new_filter = {}
        for key in self.config['filter']:
            value = self._get_value(key)
            if value is not False:
                new_filter[key] = value
        return new_filter

    def _get_value(self, key):
        value = self.config['filter'][key]
        if value is '' or value is 'False':
            return False

        if ',' in value:
            values = []
            for string in value.split(','):
                values.append(string.strip().lower())
            return set(values)

        return [value.strip().lower()]

    def _get_required_keys(self):
        value = self.config['main']['RequiredKeys']
        if ',' in value:
            values = []
            for string in value.split(','):
                values.append(string.strip().lower())
            return values

        return [value.strip().lower()]

    def _get_retained_keys(self):
        keys = []
        for key in self.config['savedstats']:
            if self.config['savedstats'].getboolean(key):
                if key not in KEYS_WITH_DICT_VALUES:
                    keys.append(key)
                else:
                    for dict_name in self.config['savedstats.%s' % key]:
                        key_list = self.config['savedstats.%s' % key][dict_name].split(',')
                        keys.append({key: key_list})
        return set(keys)
