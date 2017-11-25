import pandas as pd
from nltk.corpus import stopwords
from collections import defaultdict
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from textblob import Word
from pprint import pprint  # pretty-printer
from sklearn.metrics import  silhouette_score



cachedStopWords = stopwords.words("english")
# cachedStopWords = cachedStopWords +['congratulation'+'congrats','world','cup']
# cachedStopWords(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation

# df_data = pd.read_csv("cluster.csv")
with open("cluster.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip().lower() for x in content]
frequency = defaultdict(int)
doc_clean =[]
for text in content:
    text = re.sub("[^a-zA-Z]+", " ", text)
    text = re.sub("kabadi", "kabaddi", text)
    text = re.sub("fide", "fidel", text)
    text = re.sub("snap chat", "snapchat", text)
    text = re.sub("aliciavikanfer", "alicia vikander", text)
    text = re.sub("fidelcastro", "fidel castro", text)
    text = re.sub("andthewinnerisaliciavikander","winner is alicia vikander", text)
    text = re.sub("kabaddiworldcup","kabaddi world cup", text)
    text = re.sub("karunnair","karun nair", text)
    text = re.sub("worldcup","world cup", text)
    text = re.sub("workinggggggggg","working", text)
    line = ""
    for token in text.split():
        # normalization
        norm_word = Word(Word(token).lemmatize("v")).lemmatize()
        line = line + " " + norm_word
        if norm_word not in cachedStopWords:
            frequency[norm_word] += 1

    doc_clean.append(line)

doc_pre_processed =[]

# remove less freq words
for text in doc_clean:

    line = ''
    for token in text.split():
        # norm_word = Word(Word(token).lemmatize("v")).lemmatize()
        if frequency[token] > 1:
            line = line + " " + token
    doc_pre_processed.append(line)

pprint(doc_pre_processed)

# get freq of words
vectorizer = CountVectorizer(ngram_range=(1, 2))#stop_words=cachedStopWords
X = vectorizer.fit_transform(doc_pre_processed).toarray()
# copute proximity using cosine
cos_dist = pdist(X, 'cosine')
dists = squareform(cos_dist)
linkage_matrix = linkage(dists, "complete")

# dendogram
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    linkage_matrix,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=3.,  # font size for the x axis labels
)
plt.show()
fig.savefig('dendogram_sample.png')   # save the figure to file
plt.close(fig)

# clustering using thresh
clusters = sch.fcluster(linkage_matrix, 1.12, 'distance')#1.5
print("Clusters found: %d"%max(clusters))
# copute score
silhouette_avg = silhouette_score(X, clusters)
print("silhouette score: %f"%silhouette_avg)

# dump results
df_data = pd.DataFrame( columns=["text","preprocessed","cluster"])
df_data["text"]= content
df_data["preprocessed"] = doc_clean
df_data["cluster"] = clusters
df_data.to_csv("OP_cluster_sample.csv")



print ("done...")