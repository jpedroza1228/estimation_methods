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