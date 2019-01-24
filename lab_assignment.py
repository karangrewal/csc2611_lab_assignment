""" lab assignment """

import nltk
from nltk.corpus import brown
import numpy as np

#################################### PART 1 ###################################

################################# steps 1 & 2 #################################

# helper function
def any_alpha(s):
    for _s in s:
        if _s.isalpha():
            return True
    return False

fdist = nltk.FreqDist(w.lower() for w in brown.words())
res = fdist.most_common(5100)
i = 0
while i < len(res):
    if not any_alpha(str(res[i][0])):
        res.pop(i)
    else:
        i += 1

res = res[:5000]

w = list()
for wr, c in res:
    w.append(wr)

rg65_words = open('rg65.txt').read().strip().split('\n')
for word in rg65_words:
    if word not in w:
        w.append(word)

################################### step 3 ####################################

data = dict()
corpus = brown.words()
for i in range(1, len(corpus)):
    w1, w2 = str(corpus[i-1]).lower(), str(corpus[i]).lower()
    if w1 in w and w2 in w:
        if (w.index(w1), w.index(w2)) not in data:
            data[(w.index(w1), w.index(w2))] = 0
        data[(w.index(w1), w.index(w2))] += 1

row, col, d = list(), list(), list()
for (i, j), v in data.items():
    row.append(i)
    col.append(j)
    d.append(v)

from scipy.sparse import coo_matrix
M1 = coo_matrix((d, (row, col)), shape=(len(w), len(w)))

################################### step 4 ####################################

col_sum = M1.sum(axis=0).flatten()
row_sum = M1.sum(axis=1).flatten()
m1_sum = M1.sum()

row_plus, col_plus, ppmi = list(), list(), list()
for i in range(len(row)):
    p_x, p_y, p_xy = 1. * col_sum[0, col[i]] / m1_sum, 1. * row_sum[0, row[i]] / m1_sum, 1. * d[i] / sum(d)
    ppmi_i = max(0., np.log(p_xy / (p_x * p_y)))
    if ppmi_i > 0.:
        ppmi.append(ppmi_i)
        row_plus.append(row[i])
        col_plus.append(col[i])

M1_plus = coo_matrix((ppmi, (row_plus, col_plus)), shape=(len(w), len(w)))

################################### step 5 ####################################

from sklearn.decomposition import PCA
pca10, pca100, pca300 = PCA(n_components=10), PCA(n_components=100), PCA(n_components=300)
M2_10, M2_100, M2_300 = pca10.fit_transform(M1_plus.toarray()), pca100.fit_transform(M1_plus.toarray()), pca300.fit_transform(M1_plus.toarray())

################################### step 6 ####################################

s = list([.02, .04, .05, .06, .11, .14, .18, .19, .39, .42, .44, .44, .45, .57, .79, .85, .88, .9, .91, .96, .97, .97, .99, 1., 1.09, 1.18, 1.24, 1.26, 1.37, 1.41, 1.48, 1.5, 1.69, 1.78, 1.82, 2.37, 2.41, 2.46, 2.61, 2.63, 2.63, 2.69, 2.74, 3.04, 3.11, 3.21, 3.29, 3.41, 3.45, 3.46, 3.46, 3.59, 3.6, 3.65, 3.66, 3.68, 3.82, 3.84, 3.88, 3.92, 3.94, 3.94])

################################### step 7 ####################################

from sklearn.metrics.pairwise import cosine_similarity
pairs = open('rg65_pairs.txt').read().strip().split('\n')
pairs = [tuple(pairs[i].split(' ')) for i in range(len(pairs))]

# word-contexts
s_m1, labels = list(), list()
cos_sim = cosine_similarity(M1, M1)
for n in range(len(pairs)):
    if pairs[n][0] in w and pairs[n][1] in w:
        i, j = w.index(pairs[n][0]), w.index(pairs[n][1])
        s_m1.append(cos_sim[i, j])
        labels.append(pairs[n])

# ppmi
s_m1_plus = list()
cos_sim = cosine_similarity(M1_plus, M1_plus)
for n in range(len(labels)):
    i, j = w.index(labels[n][0]), w.index(labels[n][1])
    s_m1_plus.append(cos_sim[i, j])

# PCA 10
s_m2_10 = list()
cos_sim = cosine_similarity(M2_10, M2_10)
for n in range(len(labels)):
    i, j = w.index(labels[n][0]), w.index(labels[n][1])
    s_m2_10.append(cos_sim[i, j])

# PCA 100
s_m2_100 = list()
cos_sim = cosine_similarity(M2_100, M2_100)
for n in range(len(labels)):
    i, j = w.index(labels[n][0]), w.index(labels[n][1])
    s_m2_100.append(cos_sim[i, j])

# PCA 300
s_m2_300 = list()
cos_sim = cosine_similarity(M2_300, M2_300)
for n in range(len(labels)):
    i, j = w.index(labels[n][0]), w.index(labels[n][1])
    s_m2_300.append(cos_sim[i, j])

################################### step 8 ####################################

from scipy.stats import pearsonr
print('r(s, s_M1):         {}'.format(np.round(pearsonr(s, s_m1)[0], 3)))
print('r(s, s_M1_plus):    {}'.format(np.round(pearsonr(s, s_m1_plus)[0], 3)))
print('r(s, s_M2_PCA-10):  {}'.format(np.round(pearsonr(s, s_m2_10)[0], 3)))
print('r(s, s_M2_PCA-100): {}'.format(np.round(pearsonr(s, s_m2_100)[0], 3)))
print('r(s, s_M2_PCA-300): {}'.format(np.round(pearsonr(s, s_m2_300)[0], 3)))

#################################### PART 2 ###################################

################################### step 2 ####################################


from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

M_w2v = np.zeros((0, 300))
for word in w:
    try:
        M_w2v = np.concatenate((M_w2v, model[word].reshape(1, -1)), axis=0)
    except KeyError:
        M_w2v = np.concatenate((M_w2v, np.zeros((1, 300))), axis=0)

################################### step 3 ####################################

# cosine similarity - word2vec
s_w2v = list()
cos_sim = cosine_similarity(M_w2v, M_w2v)
for n in range(len(labels)):
    i, j = w.index(labels[n][0]), w.index(labels[n][1])
    s_w2v.append(cos_sim[i, j])

################################### step 4 ####################################

# analogy test
sem_w2v_results, sem_lsa_results = list(), list()
analogies = open('semantic_analogy.txt').read().strip().split('\n')
for i in range(len(analogies)):
    line = analogies[i].strip()
    if line.startswith(':'):
        continue
    w1, w2, w3, w4 = line.split(' ')
    w1, w2, w3, w4 = w1.lower(), w2.lower(), w3.lower(), w4.lower()
    if w1 not in w or w2 not in w or w3 not in w or w4 not in w:
        continue
    w1_w2v, w2_w2v, w3_w2v, w4_w2v = model[w1], model[w2], model[w3], model[w4]
    w1_lsa, w2_lsa, w3_lsa, w4_lsa = M2_10[w.index(w1), :], M2_10[w.index(w2), :], M2_10[w.index(w3), :], M2_10[w.index(w4), :]
    # cosine similarity
    sem_w2v_results.append(cosine_similarity((w1_w2v - w2_w2v + w4_w2v).reshape(1, -1), w3_w2v.reshape(1, -1))[0, 0])
    sem_lsa_results.append(cosine_similarity((w1_lsa - w2_lsa + w4_lsa).reshape(1, -1), w3_lsa.reshape(1, -1))[0, 0])

syn_w2v_results, syn_lsa_results = list(), list()
analogies = open('syntactic_analogy.txt').read().strip().split('\n')
for i in range(len(analogies)):
    line = analogies[i].strip()
    if line.startswith(':'):
        continue
    w1, w2, w3, w4 = line.split(' ')
    w1, w2, w3, w4 = w1.lower(), w2.lower(), w3.lower(), w4.lower()
    if w1 not in w or w2 not in w or w3 not in w or w4 not in w:
        continue
    w1_w2v, w2_w2v, w3_w2v, w4_w2v = model[w1], model[w2], model[w3], model[w4]
    w1_lsa, w2_lsa, w3_lsa, w4_lsa = M2_10[w.index(w1), :], M2_10[w.index(w2), :], M2_10[w.index(w3), :], M2_10[w.index(w4), :]
    # cosine similarity
    syn_w2v_results.append(cosine_similarity((w1_w2v - w2_w2v + w4_w2v).reshape(1, -1), w3_w2v.reshape(1, -1))[0, 0])
    syn_lsa_results.append(cosine_similarity((w1_lsa - w2_lsa + w4_lsa).reshape(1, -1), w3_lsa.reshape(1, -1))[0, 0])

sem_w2v_score, syn_w2v_score = np.round(np.mean(np.array(sem_w2v_results)), 3), np.round(np.mean(np.array(syn_w2v_results)), 3)
sem_lsa_score, syn_lsa_score = np.round(np.mean(np.array(sem_lsa_results)), 3), np.round(np.mean(np.array(syn_lsa_results)), 3)

print('average word2vec score: {} (sem)  {} (syn)'.format(sem_w2v_score, syn_w2v_score))
print('average LSA score:      {} (sem)  {} (syn)'.format(sem_lsa_score, syn_lsa_score))
