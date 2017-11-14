import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora
import re
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from textblob import Word
cachedStopWords = stopwords.words("english")
# cachedStopWords(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation

# df_data = pd.read_csv("cluster.csv")
with open("cluster.csv") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip().lower() for x in content]
frequency = defaultdict(int)
for text in content:
    text = re.sub("[^a-zA-Z]+", " ", text)
    for token in text.split():
        norm_word = Word(Word(token.lower()).lemmatize("v")).lemmatize()
        if norm_word not in cachedStopWords:
            frequency[norm_word] += 1

doc_clean = [' '.join([token for token in text.split() if frequency[token] > 2])
         for text in content]

from pprint import pprint  # pretty-printer
pprint(doc_clean)

vectorizer = CountVectorizer(stop_words=cachedStopWords)
X = vectorizer.fit_transform(content).toarray()

cos_dist = pdist(X, 'cosine')
dists = squareform(cos_dist)

linkage_matrix = linkage(dists, "single")
# dendrogram(linkage_matrix)
# create figure & 1 axis
fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis

# plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    linkage_matrix,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=3.,  # font size for the x axis labels
)
plt.show()
fig.savefig('to.png')   # save the figure to file
plt.close(fig)
clusters = sch.fcluster(linkage_matrix, 1.8, 'distance')
print("Clusters found: %d"%max(clusters))
df_data = pd.DataFrame( columns=["text","preprocessed","cluster"])
df_data["text"]= content
df_data["preprocessed"] = doc_clean
df_data["cluster"] = clusters

df_data.to_csv("OP_cluster.csv")

a=0
# Creating the term dictionary of our corpus, where every unique term is assigned an index.
# dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
# doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
#
# import  sklearn
#
# for idx,vect in enumerate(doc_term_matrix):
#     # gensim.matutils.cossim(doc_term_matrix[0], doc_term_matrix[6])
#     # sklearn.metrics.pairwise.cosine_similarity(np.asarray([1, 1, 0]), np.asarray([1, 1, 0]))