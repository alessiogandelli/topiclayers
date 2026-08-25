# tweets-to-network
```bash
pip install tweets-to-topic-network
```

This repository contains the code to 
- create networks from a set of tweets in json format (as they come from the API (RIP))
- label each tweet with a topic using BERTopic

from a set of tweets it is possibile to generate multiple networks:
- *retweet network*:  a network where the nodes are the users and the edges are the retweets
- *retweet network multilayer*: a network where the nodes are the users and the edges are the retweets, but the edges are divided in layers based on the topic of the tweet
- *temporal text network*: a bipartite network where the nodes are the tweets and the users and the edges are the interactions between them

#  Usage

import the classes
```python
from tweets_to_topic_network.data import Data_processor 
from tweets_to_topic_network.topic import Topic_modeler
from tweets_to_topic_network.network import Network_creator
```

load the json files and process them
```python
data = Data_processor(file_tweets, file_user, '22')
data.process_json() # this process the data and creates several dataframes useful for the next steps
```

Run topic modeling, takes as parameters
- dataframe: only requirement is the column 'text', 
- name: name of the dataset
- embeddedder_name: the supported embedding models are all the sentence transfers models, you can find them [here](https://www.sbert.net/docs/pretrained_models.html), and openai `text-embedding-ada-002` , `text-embedding-3-large` `text-embedding-3-small`
- path_cache: the path where to save the labeled dataframe and the model in safetensor format

returns the input dataframe with a new column with the topic of the tweet

```python
tm = Topic_modeler(data.df_original, name = data.name, embedder_name='all-MiniLM-L6-v2', path_cache = data.path_cache)
df_labeled = tm.get_topics() # this creates a new column in the dataframe with the topic of the tweet
```
Update the dataframe with the labeled topics

```python
df_retweet_labeled = data.update_df(df_labeled) # this updates the dataframe with the labeled topics
```
create the networks 
```python
nw = Network_creator(df_retweet_labeled, name = data.name, path = data.folder)
rt_network = nw.create_retweet_network() # this creates the retweet network
rt_network_ml = nw.create_retweet_ml()   # this creates the multilayer network
tt_network = nw.create_ttnetwork()       # this creates the temporal text network
```

## Qdrant integration

you have the opportunity to store the embeddings in a [qdrant](https://qdrant.tech/documentation/quick-start/) instance, you need docker, if you do not do it, nothing happens

```shell
docker pull qdrant/qdrant

```

this will create a set of folders in your current directory, to explore the embeddings stored go on [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

```shell
docker run -p 6333:6333 -p 6334:6334  -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant

```
create a dotenv file with the following variables 

```shell
OPENAI_API_KEY=your-api-key
QDRANT_URL=http://localhost:6333
```


# Input 
The input is in [jsonl](https://jsonlines.org/) format, so one JSON per line representing or a user or a tweet depending on the file. You can find an example in [this](https://github.com/alessiogandelli/tweets-to-topic-network/blob/main/data/toy.json) file. This format is the one you can get querying the Twitter API. make sure your input file is in this format.

## JSON tweets file 
The JSON tweets file contains information about all tweets involved in the conversation. Each tweet is represented as a JSON object with the following fields:

- `author`: The ID of the tweet author.
- `text`: The content of the tweet.
- `created_at`: The timestamp when the tweet was created.
- `lang`: The language of the tweet.
- `conversation_id`: The ID of the conversation the tweet belongs to (if available).
- `referenced_type`: The type of the referenced tweet (if available).
- `referenced_id`: The ID of the referenced tweet (if available).
- `mentions_name`: A list of usernames mentioned in the tweet.
- `mentions_id`: A list of IDs of users mentioned in the tweet.


## JSON users file (optional)
The user's file is used to get the username of the author of the tweet. 

- `id`: The ID of the user.
- `username`: The username of the user.


# Files created: cache and networks

## Cache
When the data is processed it will be saved in a cache folder to avoid processing the same data multiple times. The cache folder will be in the same folder as the file you are processing, i suggest you to create a folder for each set of tweets you want to process. Pickle files are for machines, csv files are for humans.
The cache folder will contain the following structure:

```shell
cache
├── data
│   ├── tweets_cop22.csv        # this is the dataframe of the tweets 
│   ├── tweets_cop22.pkl
│   ├── tweets_cop22_topics.pkl # this is the labeled dataframe with topics
│   ├── users_cop22.csv         # this is the dataframe of the users
│   └── users_cop22.pkl
└── tm
    └── all-MiniLM-L6-v2            # the folder for this embedding model
        ├── embeddings_cop22.pkl    # the embeddings of the tweets 
        ├── model_cop22             # the model in safetensor format
        │   ├── config.json
        │   ├── ctfidf.safetensors
        │   ├── ctfidf_config.json
        │   ├── topic_embeddings.safetensors
        │   └── topics.json
        └── topics_cop22.csv    # topics, keyworkds and representative tweets

```

## Networks
The networks will be saved in the networks folder in the same folder as the file you are processing. Note that there are two different retweet network, one is generated from the temporal text network, the other directly from the tweets, there could be differences like self edges, still have to figure out which one is better. The networks folder will contain the following structure:

```shell
├── networks
│   ├── cop22_retweet.gml                   # retweet network
│   ├── cop22_retweet_ml.gml                # retweet network multilayer
│   ├── cop22_retweet_network_ml.gml       # retweet network multilayer from projection
│   ├── cop22_ttt.gml                   # temporal text network
│   └── projected                   # each layer (topic) of the multilayer network
│       ├── cop22__prj_-1.0.gml
│       ├── cop22__prj_0.0.gml
│       ├── cop22__prj_1.0.gml
│       ├── cop22__prj_10.0.gml
│       ├── cop22__prj_11.0.gml


```

# Classes
The pipeline is divided into three classes, each of them is independent and can be used separately.

- *Data_processor*: takes the two json files (or just one) and creates a pandas dataframe with all the tweets . It also creates a cache of the dataframe in csv and pickle format.

- *Topic_modeler*: takes a dataframe and labels the tweets with a topic using BERTopic. It also creates a cache of the labeled dataframe in csv and pickle format.

- *Network_creator*: takes a dataframe of tweets and creates a network from it. It can create a retweet network, a temporal text network, and a multilayer network. The networks are saved in gml format.


## Data Processor
The first step is to initialize the pipeline class giving the two files as input and the name of the COP. (file_user is optional, if not present the username will be the user id)

```python
data = Data_processor(file_tweets=file_tweets, file_user=file_user, n_cop='22')
data.process_json()

```
### Under the hood
Let's see what happen when you run the`process_json()` function, firstly, if the files are present in the cache folder, it loads them, otherwise, it follows this pipeline:.
- load_users_json: loads the json file of the users into a dataframe 
- load_tweets_json: loads the json file of the tweets into a dataframe discarding the tweets with attachments
- save dataframes: both in pickle (for machines) and csv(for humans) format in the cache folder

### After the topic modeling
There is the possibility to update the dataframe with the labeled topic, so that the object data contains all the data needed, it gets the labeled dataframe of the original tweets (the topic modeling is not run on retweets to save save time and energy), and propagate the topic also to the retweets of the original tweets.

```python  
df_retweet_labeled = data.update_df(df_labeled)

```
## Topic Modeler
Then we can get the topics of the tweets using BERTopic, this creates a new column in the dataframe with the topic of the tweet. The topics are saved in a pickle file for caching. In [this](https://github.com/alessiogandelli/topic-modeling-eval) repository I conclude that BERTopic is the best topic modeling algorithm for this scenario. You can choose the embedder you prefer

```python
tm = Topic_modeler(data.df_original, name = data.name, embedder_name='all-MiniLM-L6-v2', path_cache = data.path_cache)
df_labeled = tm.get_topics()


```
### label topics (todo)
we use openai gpt3.5turbo model to label the topic using the putput of the cTFIDF and some representative tweets. if you want to use openai api, you need an env variable with the api key, otherwise it will not work. 
OPENAI_API_KEY=sk-xxxxx

## Network Creator
Finally we can create the networks using the full dataframe with topic labels. there are 3 possible types of network you can create

```python
nw = Network_creator(df_retweet_labeled, name = data.name, path = data.folder)
```
### Temporal Text Network
The function creates a [temporal text network](https://appliednetsci.springeropen.com/articles/10.1007/s41109-018-0082-3) and two dictionaries that map the tweet id to its text and the couple (user, tweet) to the time of the tweet. The network is saved in gml format in the `networks` folder.

```python
nw.create_ttnetwork(project = True)
```

In this example we can see how a network is built, the small nodes are the tweets, while the other are the authors. The color of the tweets corresponds to the topic of the tweet. The edges are of three types:
- user-tweet: if the user has tweeted the tweet
- tweet-user: if the tweet mention the user
- tweet-tweet: if the tweet retweets the other

![full network](https://github.com/alessiogandelli/tweets-to-topic-network/blob/main/data/full_network.png)



#### Network projection
if the project parameter is set to True the network is also projected into a retweet network (this is redundant, you can create a retweets network without passing from the ttn) and a network of retweets for each topic :

In the temporal text network, we have two sets of nodes and we want only one: the authors. To achieve this I used a hybrid approach between iteration and recursion. First, it iterates over all the users, so for each user, the goal now is to create all the edges between this user and others. 

For each user, we iterate over all its tweets recursively searching for the end of the chain. There are a few cases:
- It is an original tweet with no mentions: do nothing 
- It is an original tweet mentioning other users: create an edge between the user and the mentioned users
- It is a retweet: create an edge between the user and the author of the original tweet.

In this process are also involved the topics of the tweets, so while projecting it, a network is created for each topic, and the edges are added to the corresponding network. The networks are saved in gml format in the `networks/projected` folder.

The TTN is a very powerful tool, but it requires a lot of computational power and time. so if you do not need all the information contained, you can create the simple retweet network using the `retweet_network()` or the  `retweet_network_ml()` depending if you are interested in the multilayer network or not.

### Retweet Network

In this network the nodes are the users and the edges are the retweets, the network is saved in gml format in the `networks` folder.

```python
nw.create_retweet_network()
```

### Retweet Network Multilayer


multiples retweets network created at topic level stacked together

This is an example of how it looks like 

![multilayer network](https://github.com/alessiogandelli/tweets-to-topic-network/blob/main/data/projected_topics_ml.png)





#  How to run it

suggestion: create a folder for each set of tweets you want to process, in our case one for each COP. Inside this folder there should be the two jsonl files, one for the tweets and one for the users. The script will create the following directories:

- `cache`: contains the csv and pickle files of the intermediate steps of the pipeline, from the cleaned dataset in csv to the label of the topics.

- `networks`: contains the gml files of the temporal text network




# A Real Example

Processing around 400k tweets with the hashtag #cop22, this took around 3 hours on 2018 macbook air.

- json tweets: 878 MB
- csv/pkl tweets: 236 MB
- retweet gml network: 234 MB
- projected gml network: 19 MB

## cop 26 
macbook pro 14 

- json tweets: 14.3 GB (requires a lot of ram)
- csv/pkl tweets: 1.79 GB
- temporal text network: 900 MB
- projected multilayer: 84 MB

2 h for processing the json file 
4 h for extracting topics 
18 h for creating the networks 


# Troubleshooting
since there is a dependency on uunet which uses older cmake version you could have problems installing it with python 3.12, but with python 3.11 it should work fine.