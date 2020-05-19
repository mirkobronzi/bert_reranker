import pickle 
import numpy as np
import matplotlib.pyplot as plt
import pdb 
import sklearn.manifold as sktm

np.random.seed(2)
data = pickle.load(open('embs.pkl', 'rb'))

embeddings = (data['embs']) 
vals = data['vals']
topics = data['topics']
unique_topics = list(np.unique(topics))

X_embedded = sktm.TSNE(n_components=2, verbose=1).fit_transform(np.concatenate(embeddings))

#plt.imshow(embeddings[:30].transpose())
N_topics = len(np.unique(topics))

colors = 'rbkmcrbkmbcrkmc'
markers = 'xo^svosvxs^'

handles = [1 for i in range(N_topics)]
visited = [] 
for emb, val, topic in zip(X_embedded, vals, topics):
    try:
        topic_ind = unique_topics.index(topic)
    except:
        topic_ind = -1

    if topic_ind not in visited:
        plt.plot(emb[0], emb[1], colors[topic_ind] + markers[topic_ind], label=unique_topics[topic_ind])
    else: 
        plt.plot(emb[0], emb[1], colors[topic_ind] + markers[topic_ind])

    visited.append(topic_ind)

plt.legend()

plt.show()
pdb.set_trace()
