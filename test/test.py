# %%
import pandas as pd

# %%

print('ciao')

# %%
import numpy as np
import pandas as pd
from tweets_to_topic_network.network import Network_creator

df = pd.DataFrame({
    'id' : ['t1', 't2', 't3', 't4', 't5', 't6', 't7'],
    'author': ['a', 'b', 'c', 'c', 'a', 'd', 'd'],
    'text': ['tweet1', 'tweet2', 'tweet3', 'tweet4', 'tweet5', 'tweet6', 'tweet7'],
    'date': pd.date_range(start='01-01-2022', periods=7).strftime('%Y-%m-%d').tolist(),
    'lang': ['en', 'en', 'en', 'en', 'en', 'en', 'en'],
    'referenced_type': ['original', 'retweet', 'retweet', 'original', 'retweet', 'retweet', 'original'],
    'referenced_id': [ None, 't1', 't2', None, 't7', 't7', None],
    'topic' : [1,1,1,1,2,2,2],
    'mentions_name': [['b'], [], [], [], [], [], []]
    
}).set_index('id')

# %%Print the DataFrame
nw = Network_creator(df, name = 'test', path = 'data')
G = nw.create_retweet_network()


nw.create_ttnetwork()
nw.create_retweet_ml()
nw.create_retweet_network()
# %%



import matplotlib.pyplot as plt
import networkx as nx
G = nw.ttn[0]
pos = nx.planar_layout(G)
nx.draw_networkx_nodes(G, pos, node_color=[('red' if node[1]['bipartite'] == 1 else 'green') for node in G.nodes(data=True)])
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos)
plt.show()
# %%





# %%
