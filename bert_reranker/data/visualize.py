# import pickle
#
# import matplotlib.pyplot as plt
# import numpy as np
# import sklearn.manifold as sktm
#
# np.random.seed(2)
# gt_questions = pickle.load(open("GT_QUESTION_embs.pkl", "rb"))
# questions = pickle.load(open("QUESTION_embs.pkl", "rb"))
#
# embeddings = questions["embs"]
# N_questions = len(questions["embs"])
#
# for gt_emb in gt_questions["embs"]:
#     embeddings.append(gt_emb)
#
# topics = questions["topics"]
#
# # find the topics for gt_questions
# for gt_q in gt_questions["gt_questions"]:
#     ind = questions["gt_questions"].index(gt_q)
#     topics.append(questions["topics"][ind])
#
# unique_topics = list(np.unique(topics))
#
# X_embedded = sktm.TSNE(n_components=2, verbose=1).fit_transform(
#     np.concatenate(embeddings)
# )
#
# N_topics = len(np.unique(topics))
#
# colors = "rbkmcrbkmbcrkmc"
# markers = "xo^svosvxs^"
#
# visited = []
# for i, (emb, topic) in enumerate(zip(X_embedded, topics)):
#     try:
#         topic_ind = unique_topics.index(topic)
#     except:
#         topic_ind = -1
#
#     if i < N_questions:
#         if topic_ind not in visited:
#             plt.plot(
#                 emb[0],
#                 emb[1],
#                 colors[topic_ind] + markers[topic_ind],
#                 label=unique_topics[topic_ind],
#             )
#         else:
#             plt.plot(emb[0], emb[1], colors[topic_ind] + markers[topic_ind])
#     else:
#         plt.plot(
#             emb[0],
#             emb[1],
#             colors[topic_ind] + markers[topic_ind],
#             markersize=13,
#             linestyle="dashed",
#             linewidth=1,
#         )
#
#     visited.append(topic_ind)
#
# plt.legend()
#
# plt.show()
