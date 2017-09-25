# Retrieves data from the public stash API

import requests
import json
BASE_API_URL = 'http://www.pathofexile.com/api/public-stash-tabs'


class Retriever:
    def retrieve(self, next_id):
        payload = {}
        if self.next_id:
            payload = {'id': self.next_id}

        r = requests.get(BASE_API_URL, params=payload)
        json_data = json.loads(r.text.lower())
        next_id = self._decode_next_id(json_data)
        return (json_data['stashes'], next_id)

    def _decode_next_id(self, json_data):
        # Indicates that there is a next page of data to retrieve
        # If not true, the next update has not yet been released
        if len(json_data['stashes']) > 0:
            return json_data['next_change_id']
        else:
            print('Could not yet get next id. Reached end of updates for now.')
            return None
