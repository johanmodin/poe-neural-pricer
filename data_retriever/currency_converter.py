import requests
import json

BASE_API_URL = 'http://poe.ninja/api/Data/GetCurrencyOverview'

SHORTHAND_TRANSLATIONS = {
'exalted orb' : ['exa', 'ex'], 'chaos orb': ['chaos', 'c'], 'orb of fusing': ['fuse'],
'orb of alchemy': ['alch'], 'orb of alteration': ['alt'], 'gemcutter\'s prism': ['gcp'],
'chromatic orb': ['chrom'], 'jeweller\'s orb': ['jew'], 'orb of chance': ['chance'],
'cartographer\'s chisel': ['chisel'], 'orb of scouring': ['scour'],
'blessed orb': ['blessed'], 'orb of regret': ['regret'], 'regal orb': ['regal'],
'divine orb': ['divine'], 'vaal orb': ['vaal']
}

# Defines when the price gets unreasonable enough for us to skip it
# Value is in chaos orbs. This way we remove some of the outliers.
MAX_VALUE_CUT_OFF = 20000

class CurrencyConverter:
    def __init__(self, league='Harbinger'):
        self.rates = self._retrieve_rates(league)

    def convert(self, note):
        data = note.split(' ')
        if len(data) < 3:
            return -1
        amount = data[1]
        shorthand = data[2]
        try:
            float(amount)
        except ValueError:
            return -1
        if shorthand not in self.rates:
            return -1
        if float(amount)*float(self.rates[shorthand]) >= MAX_VALUE_CUT_OFF:
            return -1
        return float(amount)*float(self.rates[shorthand])

    def _retrieve_rates(self, league):
        r = requests.get(BASE_API_URL, params={'league': league})
        json_data = json.loads(r.text.lower())
        return self._extract_rates(json_data)

    def _extract_rates(self, json_data):
        rates = {}
        for currency in json_data['lines']:
            if currency['currencytypename'] in SHORTHAND_TRANSLATIONS:
                for shorthand in SHORTHAND_TRANSLATIONS[currency['currencytypename']]:
                    rates[shorthand] = currency['chaosequivalent']
        for shorthand in SHORTHAND_TRANSLATIONS['chaos orb']:
            rates[shorthand] = 1
        return rates
