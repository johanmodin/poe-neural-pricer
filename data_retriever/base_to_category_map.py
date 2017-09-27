import requests
import json

BASE_URL = 'http://poe.trade'
# Gets a dictionary mapping from base item type -> item category
# For example 'Battle Plate': 'Body Armour'
# We need this as the base item type is visible in the PoE stash API whilst
# its category is not.
def get_dict():
    print('Fetching base item -> item category dict from poe.trade')
    r = requests.get(BASE_URL)

    # We're selecting sort of a dictionary from poe.trade which contains all the
    # item classes and their corresponding base items
    t = r.text.split('var items_types = ')[1].split(';')[0].lower()

    # We make a dictionary out of it which now contains the base items as keys
    # and item classes as values
    j = json.loads(t)

    # To get it the other way around we inverse the dictionary
    item_mapping = dict((v,k) for k in j for v in j[k])
    return item_mapping
