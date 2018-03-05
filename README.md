# poe-neural-pricer 

###### Not currently maintained.

This application collects active Path of Exile trade listings for specified items with the goal of being able to price your items instantaneously. 

The idea behind the project was to gather enough data on items and their prices to be able to train a neural network model to learn what combinations of item attributes and their respective values that made an item valuable. The end goal for the project was to be able to price items. Especially, it was meant to be able to predict rare items' prices as they are usually very hard to price due to the many niche uses that rare items may allow for specific builds.

Currently, the network does not render in very useful pricing results. My opinion on how to fix this is detailed below.

### Details
The application downloads specified items that are listed on PoE. The items' prices are then converted to a common currency, namely Chaos orbs, by downloading the current currency rates for the specified league. 

The items attributes are encoded to integers with a labeller which are used to place values at the respective index in a feature array that is generated for the item. The feature arrays are saved along with the target prices as [(X,y)] numpy arrays. This data may then be fed to the neural network.

Future improvements that could make this tool more useful: 
* Making the data sparser to significantly decrease the feature length as it currently may be a very large 1D array.
* Use models with, for the task, better architectures than the current one (which is just sort of a default model). Perhaps  convolutional layers could be used if the feature arrays were differently arranged.
* "Ignore item listings that are older than x"-option. Currently, the db contains listings that are years old and which likely corrupts the data.
* Unify how item attributes are encoded. Some are simply true/false whilst others contain values, some with several values, etc.
* Better means of balancing data

### Usage
For collecting the items specified in config.ini, create a Controller c and run c.collect(iterations)
For encoding the collected items, create a Controller c and run c.encode(workers)
To train a model, specify the model in /network/train_network.py and your data_dir and save_dir as well as the parameters for how many files are to be concurrently held in memory, etc. Run t = Trainer('your_model_name') or t = Trainer(loading_model='my_already_trained_model') to start the process of training.
