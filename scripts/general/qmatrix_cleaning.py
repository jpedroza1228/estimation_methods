import pandas as pd
import numpy as np
from pyhere import here
from janitor import clean_names
from great_tables import GT as gt

df = pd.read_csv(here('q_matrix/quiz5_q.csv'))

df = df.loc[~df['item'].isin([22, 23])]

df['item'] = np.select([(df['item'] == 24),
                        (df['item'] == 25),
                        (df['item'] == 26)],
                       [22, 23, 24],
                       default=df['item'])

df['item'] = np.select(
    [df['item'] == item for item in np.arange(27, 38)],
    np.arange(25, 36),
    default=df['item']
)

gt(df)

# df.to_csv(here('q_matrix/quiz5_q_updated.csv'))




retake = pd.DataFrame({'question': [[1] * 5 + [2] * 9 + [3] * 6 + [4] * 5 + [5] * 4 + [6] * 6],
                       'item': [np.arange(1, 36, 1)],
                       'Expectations of functions': [[1] * 5 + [0] * 30],
                       'Definition of variance': [[1] * 5 + [0] * 20 + [1] * 4 + [0] * 6],
                       'Properties of expectation': [[0] * 5 + [1] * 9 + [0] * 6 + [1] * 5 + [0] * 10],
                       'Expectations of famous distributions': [[0] * 14 + [1] * 11 + [0] * 10],
                       'Properties of Variance': [[0] * 29 + [1] * 6]})
retake = retake.explode(['question', 'item', 'Expectations of functions', 'Definition of variance', 'Properties of expectation', 'Expectations of famous distributions', 'Properties of Variance'])

retake.to_csv(here('q_matrix/retake5_q_5attr.csv'))