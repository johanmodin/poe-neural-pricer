# poe-scraper
This application collects active Path of Exile trade listings for specified items. 

The idea behind the project was to gather enough data on items and their prices to be able to train a neural network model to learn what combinations of item attributes and their respective values that made an item valuable. The end goal for the project was to be able to price items.

###### It is not currently being updated.

### Details
The application downloads specified items that are listed on PoE. The items' prices are then converted to a common currency, namely Chaos orbs, by downloading the current currency rates for the specified league. The items attributes are then encoded with a labeller and saved as a numpy array consisting of categories along with the target prices. This data may then be fed to the neural network.
