# poe-scraper
This application collects active Path of Exile trade listings for specified items. 

The idea behind the project was to gather enough data on items and their prices to be able to train a neural network model to learn what combinations of item attributes and their respective values that made an item valuable. The end goal for the project was to be able to price items. Especially, it was meant to be able to predict rare items' prices as they are usually very hard to price due to the many niche uses that rare items may allow for specific builds.

###### It is not currently being updated.

### Details
The application downloads specified items that are listed on PoE. The items' prices are then converted to a common currency, namely Chaos orbs, by downloading the current currency rates for the specified league. 

The items attributes are then encoded to integers with a labeller and saved as a numpy array consisting of categories along with the target prices. This data may then be fed to the neural network.

Future improvements that could make this tool more useful would be: 
* Making the data sparser to decrease the feature length as it currently may be a very large 1D array 
* Use models with, for the task, better architectures than the current one (which is just sort of a default model)
* "Ignore item listings that are older than x"-option. Currently, the db contains listings that are years old and which likely corrupts the data.
* Unify how attributes are encoded. Some are simply true/false whilst others contain values, some with several values, etc.
