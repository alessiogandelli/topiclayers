#%%
import pandas as pd
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import re
import json

from openai import OpenAI
client = OpenAI()


path = '/Users/alessiogandelli/data/cop_merged/en/topics_english.csv'
path_data = '/Users/alessiogandelli/data/cop_merged/en/tweets_labeled_en_merged.pkl'

df = pd.read_csv(path)
df = df.set_index('Topic')

df_tweets = pd.read_pickle(path_data)

# %%

def topic_distribution(n_topics = 30):

    topics = df_tweets['topic'].value_counts()
    topic_perc = topics / sum(topics)
    last_topics_sum = pd.Series(sum(topic_perc[n_topics:]))


    topics = pd.concat([topic_perc[:n_topics], last_topics_sum]).to_list()


    data = topics[1:]

    # Labels for each segment (just a simple index)
    labels = [f'Segment {i+1}' for i in range(len(data))]

    # Colors for each segment
    colors = sns.color_palette("tab10", len(data))

    # Create a DataFrame to structure the data for Seaborn
    df = pd.DataFrame({
        'Category': labels,
        'Value': data,
        'Colors': colors
    })

    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(5, 10))


    # Bottom values for stacking
    bottom = 0
    for i in range(len(df)):
        sns.barplot(x=['topics'], y=[df['Value'][i]], color=df['Colors'][i], bottom=bottom, ax=ax)
        bottom += df['Value'][i]

    ax.bar('outliers', topics[0], color='lightblue', alpha=0.5)


    # Add labels and title

    ax.set_ylabel('% of tweets')
    ax.set_title('topics distribution')

    ax.text(0, 0.3,f'Topics from {n_topics} to 733', color='black', ha='center')
    ax.text(1, 0.3, 'outliers', color='black', ha='center')
    ax.yaxis.set_major_locator(plt.MaxNLocator(30))  # Adjust the number as needed



    # Show the plot
    plt.show()

topic_distribution(n_topics=30)
topic_distribution(n_topics=40)
topic_distribution(n_topics=50)
# %%

df_first = df[1:31]



filtered_tweets = df_tweets[df_tweets['topic'].isin(df_first.index)]

print(len(filtered_tweets))

# %%
filtered_tweets
# %%



# %%
#%%

topic_repr = {}
for n_topic in df_first.index:

    words = df_first.loc[n_topic]['Representation']

    tweets = filtered_tweets[filtered_tweets['topic'] == n_topic].sample(4)['text'].tolist()

    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    tweets = [url_pattern.sub('', text) for text in tweets]


    system_prompt = """I want you to act as a tweet labeler, you are given representative words
        from a topic and four representative tweets, give more attention to the words, all the tweets are related to climate change, and COP, no need to mention it, detect subtopics.
        and avoid hashtags, reply in a json format including a short and a long label"""

    user_prompt = f"""
        which is a good label for the topic containing the words [{words}], here you are tweets to help you:
        [{tweets}]"""



    response = client.chat.completions.create(
    model="gpt-4o",
    response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    )

    topic = json.loads(response.choices[0].message.content)
    topic['tweets'] = tweets

    topic_repr[n_topic] = topic




#%%

df_labels = pd.DataFrame().from_dict(topic_repr).T
df_labels['words'] = df_first['Representation']





# topic by cop
# %%
row = df_labels.loc[0]

# Print information in a structured way
print(f"Topic 0: {row['short_label']}")
print(f"|- Long Label: {row['long_label']}")
print(f"|- Keywords: {row['words']}")

# Print the first few tweets
print("|- Tweets:")
for i in range(3):  # Assuming you want to print the first 3 tweets
    print(f"|-- {row['tweets'][i]}")

# %%
