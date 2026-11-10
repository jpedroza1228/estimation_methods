import pandas as pd
import numpy as np
import os
from pyhere import here
from pathlib import Path
from great_tables import GT as gt

folder_path = here('fit_metrics/output_Boting')

csv_files = folder_path.glob("*.csv")

dfs = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(dfs, ignore_index=True).drop(columns = 'Unnamed: 0')
combined_df.sort_values('rmse').to_csv(f'{folder_path}/compare_model_fit.csv')