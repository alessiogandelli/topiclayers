# %%
import importlib
from tweets_to_topic_network.data import Data_processor 
from tweets_to_topic_network import topic
importlib.reload(topic)
from tweets_to_topic_network.topic import Topic_modeler
from tweets_to_topic_network.network import Network_creator

n_cop = 'cop22'

#file_tweets = '/Users/alessiogandelli/data/' + n_cop + '/' + n_cop + '.json'
#file_user = '/Users/alessiogandelli/data/' + n_cop + '/users_'+ n_cop+'.json'
#file_tweets= '/Volumes/NEO storage/data/climate_network/test/sample.json'
file_tweets = '/Users/alessiogandelli/dev/uni/tweets-to-topic-network/data/cop22.json'
# file_user = '/Users/alessiogandelli/dev/uni/tweets-to-topic-network/data/toy_users.json'

#%%
data = Data_processor(file_tweets, n_cop = n_cop)
data.process_json()


#%%
tm = Topic_modeler(data.df_original, name = data.name, embedder_name='all-MiniLM-L6-v2', path_cache = data.path_cache)
df_labeled = tm.get_topics()


#%%
df_retweet_labeled = data.update_df(df_labeled)


#%%
nw = Network_creator(df_retweet_labeled, name = data.name, path = data.folder)
G = nw.create_retweet_network()
nw.create_ttnetwork()
nw.create_retweet_ml()








#%%
#p.project_network(path='/Volumes/boot420/Users/data/climate_network/test/networks/sampleretweets.gml', title='retweets')


