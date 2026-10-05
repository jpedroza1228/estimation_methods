import pandas as pd
import numpy as np
from pyhere import here
from janitor import clean_names

q = pd.read_csv(here('q_matrix/quiz5_q_updated.csv')).clean_names(case_type = 'snake').drop(columns = 'unnamed_0')
q['item'] = ['item' + str(i) for i in q['item']]

# if no testlet, run this code
q = q.drop(columns = 'question')
q = q.drop(columns = 'total_variance')
q.head()

alpha = pd.DataFrame([(a, b, c, d, e, f, g) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2) for d in np.arange(2) for e in np.arange(2) for f in np.arange(2) for g in np.arange(2)])
      
alpha = alpha.rename(columns = {0: 'attr1',
                                1: 'attr2',
                                2: 'attr3',
                                3: 'attr4',
                                4: 'attr5',
                                5: 'attr6',
                                6: 'attr7'})

alpha['class_label'] = alpha.astype(int).astype(str).agg(''.join, axis = 1)
alpha = alpha.reset_index()
alpha['index'] = alpha['index'] + 1

alpha.head()

q = q.drop(columns = 'item')
alpha = alpha.drop(columns = ['index', 'class_label'])

print(q.shape)
print(alpha.shape)

alpha = alpha.to_numpy()
q = q.to_numpy()


q = np.array([
    [1, 0, 0],  # Item 1: Requires Skill A
    [1, 1, 0],  # Item 2: Requires Skills A & B
    [0, 1, 1],  # Item 3: Requires Skills B & C
    [1, 1, 1]   # Item 4: Requires Skills A, B, & C
])

# Define Alpha matrix (5 students x 3 attributes)
alpha = pd.DataFrame([(a, b, c) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2)])

eta = np.zeros((q.shape[0], alpha.shape[0]), dtype = int)

for i in range(q.shape[0]):
  for c in range(alpha.shape[0]):
    eta[i,c] = np.prod(alpha[c, :] ** q[i, :])

eta

dino_eta = np.zeros((q.shape[0], alpha.shape[0]), dtype = int)
for i in range(q.shape[0]):
  for c in range(alpha.shape[0]):
    dino_eta[i,c] = 1 - np.prod((1 - alpha[c, :]) ** q[i, :])

[eta[i, :] == dino_eta[i, :] for i in range(q.shape[0])]