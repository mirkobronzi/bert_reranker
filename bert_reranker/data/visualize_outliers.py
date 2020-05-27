# import pickle
#
# import matplotlib.pyplot as plt
# import numpy as np
# import sklearn.manifold as sktm
# import sklearn.neighbors as skn
#
# np.random.seed(2)
# gt_questions = pickle.load(open("GT_QUESTION_embs.pkl", "rb"))
# questions = pickle.load(open("QUESTION_embs.pkl", "rb"))
#
# dataset_ood = "samasource_ood"
# if dataset_ood == "natqdev":
#     ood_questions = pickle.load(open("natqdev_embs.pkl", "rb"))
# elif dataset_ood == "healthtapdev":
#     ood_questions = pickle.load(open("healthtapdev_embs.pkl", "rb"))
# elif dataset_ood == "samasource_ood":
#     ood_questions = pickle.load(open("samasourceood_embs.pkl", "rb"))
#
#
# ood_embeddings = np.concatenate(ood_questions["embs"])
#
# id_embeddings = np.concatenate(questions["embs"])
# N_in = id_embeddings.shape[0]
#
# all_embeddings = np.concatenate([id_embeddings, ood_embeddings], axis=0)
#
# mode = "novelty"
#
# colors = "byckmbyckm"
# markers = "xov^s><v^s>"
#
# if mode == "visualize":
#
#     emb = sktm.TSNE(n_components=2, verbose=1).fit_transform(all_embeddings)
#
#     visited = []
#     plt.scatter(
#         emb[:N_in, 0], emb[:N_in, 1], c="r", marker="o", label="covid_quebec_faq"
#     )
#     if dataset_ood == "samasource_ood":
#         emb_ood = emb[N_in:, :]
#         for i, (an) in enumerate(np.unique(ood_questions["annotations"])):
#             inds = np.array(ood_questions["annotations"]) == an
#             plt.scatter(
#                 emb_ood[inds, 0],
#                 emb_ood[inds, 1],
#                 c=colors[i],
#                 marker=markers[i],
#                 label=an,
#             )
#     else:
#         plt.scatter(emb[N_in:, 0], emb[N_in:, 1], c="b", marker="x", label=dataset_ood)
#
#     plt.legend()
#
#     plt.show()
# else:
#     clf = skn.LocalOutlierFactor(n_neighbors=4, novelty=True)
#
#     clf.fit(id_embeddings)
#     ys_id = clf.predict(id_embeddings)
#
#     # inds = (np.array(ood_questions['annotations']) == 'NOT_COVERED')
#     ys_ood = clf.predict(ood_embeddings)
#
#     ys_ood[ys_ood == -1] = 0
#
#     print("accuracy is {}".format((1 - ys_ood.mean()) * 100))
#
#     with open("results_samasource.txt", "w") as fl:
#         for i, (qst, an) in enumerate(
#             zip(ood_questions["questions"], ood_questions["annotations"])
#         ):
#             fl.write(qst + "\n")
#             fl.write("Samasource annotation : {} \n".format(an))
#             if ys_ood[i] == 1:
#                 fl.write("Prediction : In distribution \n\n")
#             else:
#                 fl.write("Prediction : Out of distribution \n\n")
