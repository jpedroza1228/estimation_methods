import pandas as pd
import numpy as np
import matplotlib
import plotnine as pn
from janitor import clean_names
from great_tables import GT as gt
import matplotlib.pyplot as plt
from pyhere import here
import arviz_base as azb
import arviz_plots as azp
import arviz_stats as azs
from pathlib import Path
import tempfile
from cmdstanpy import CmdStanModel
import joblib
import os

pd.set_option('display.max_columns', None)
matplotlib.rcParams.update({'savefig.bbox': 'tight'})
pn.theme_set(pn.theme_classic())

model_path = 'data/joblib_models/mle_no_inform_prior_no_testlet_5attr_dino.joblib'

data_directory = 'data/stat_134_2025_quiz5.csv'
q_directory = 'q_matrix/quiz5_q_5attr.csv'

threshold = .8
model_type = 'DINO'

attr1 = 'expect_fun'
attr2 = 'def_var'
attr3 = 'prop_expect'
attr4 = 'expect_famous_dist'
attr5 = 'prop_var'

def q_lower(x):
    return x.quantile(.025)
  
def q_upper(x):
    return x.quantile(.975)

df = pd.read_csv(here(f'{data_directory}')).drop(columns = 'Unnamed: 0')
q = pd.read_csv(here(f'{q_directory}')).clean_names(case_type = 'snake').drop(columns = 'unnamed_0')
q['item'] = ['item' + str(i) for i in q['item']]

# if no testlet, run this code
q = q.drop(columns = 'question')

df = df.drop(columns = 'item35')
df.head()

q = q.loc[q['item'] != 'item35']
q.head()

print(df.head())
print(q.head())


alpha = pd.DataFrame([(a, b, c, d, e) for a in np.arange(2) for b in np.arange(2) for c in np.arange(2) for d in np.arange(2) for e in np.arange(2)])
      
alpha = alpha.rename(columns = {0: 'attr1',
                                1: 'attr2',
                                2: 'attr3',
                                3: 'attr4',
                                4: 'attr5'})

alpha['class_label'] = alpha.astype(int).astype(str).agg(''.join, axis = 1)
alpha = alpha.reset_index()
alpha['index'] = alpha['index'] + 1

print(alpha.head())
print(alpha.shape)

xi = np.zeros((q.shape[0], alpha.shape[0]), dtype = int)

if model_type == 'DINO':
  # DINO model
  for i in range(q.drop(columns = 'item').shape[0]):
    for c in range(alpha.drop(columns = ['index', 'class_label']).shape[0]):
      xi[i,c] = 1 - np.prod((1 - alpha.drop(columns = ['index', 'class_label']).to_numpy()[c, :]) ** q.drop(columns = 'item').to_numpy()[i, :])

elif model_type == 'DINA':
  # DINA model
  for i in range(q.drop(columns = 'item').shape[0]):
    for c in range(alpha.drop(columns = ['index', 'class_label']).shape[0]):
      xi[i,c] = np.prod(alpha.drop(columns = ['index', 'class_label']).to_numpy()[c, :] ** q.drop(columns = 'item').to_numpy()[i, :])

stan_dict = {
  'J': df.drop(columns = 'anon_id').shape[0],
  'I': df.drop(columns = 'anon_id').shape[1],
  'C': alpha.drop(columns = ['class_label', 'index']).shape[0],
  'K': q.drop(columns = 'item').shape[1],
  'Y': df.drop(columns = 'anon_id').to_numpy(),
  'Q': q.drop(columns = 'item').to_numpy(),
  'alpha': alpha.drop(columns = ['class_label', 'index']).to_numpy(),
  'xi': xi
}

model, fit = joblib.load(here(model_path))

fitdf = pd.DataFrame([fit.stan_variables()])

nu_df = pd.DataFrame(
    {'name': [f'nu{i}' for i in np.arange(1, len(fitdf['nu'].item()) + 1)],
      'value': fitdf['nu'].item(),
      'class_label': alpha['class_label']}
)

nu_df

pn.ggplot.show(
  pn.ggplot(nu_df,
            pn.aes('class_label', 'value'))
  + pn.geom_col(color = 'black',
                fill = 'seagreen')
  + pn.coord_flip()
)


slip_df = pd.DataFrame(
    {'name': [f'slip{i}' for i in np.arange(1, len(fitdf['slip'].item()) + 1)],
      'value': fitdf['slip'].item()}
)

pn.ggplot.show(
  pn.ggplot(slip_df,
            pn.aes('name', 'value'))
  + pn.geom_point(color = 'seagreen')
  + pn.coord_flip()
)


guess_df = pd.DataFrame(
    {'name': [f'guess{i}' for i in np.arange(1, len(fitdf['guess'].item()) + 1)],
      'value': fitdf['guess'].item()}
)

pn.ggplot.show(
  pn.ggplot(guess_df,
            pn.aes('name', 'value'))
  + pn.geom_point(color = 'seagreen')
  + pn.coord_flip()
)



fitdf.columns
fitdf['eta'].item()
fitdf['prob_joint'].item()
fitdf['log_lik'].item()
fitdf['prob_attr_class'].item()

fitdf['prob_resp_class'].item()
fitdf['prob_resp_attr'].item()
fitdf['y_rep'].item()


y = stan_dict['Y']
rep_y = fitdf['y_rep'].item()

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y.flatten(), rep_y.flatten()))
rmse

log_lik = fitdf['log_lik'].item().mean()

num_stu = rep_y.shape[0]
num_item = rep_y.shape[1]
num_lat_class = alpha.shape[0]
num_param = (2 * num_item) + (num_lat_class - 1)

bic = -2 * log_lik + num_param * np.log(num_stu)
sabic = -2 * log_lik + num_param * np.log((num_stu + 2) / 24)


# CHANGE THE MODEL NAME
model_name_save = 'dino_no_testlet_5attr_mle'

# 5. Construct the metrics DataFrame
fit_metrics = pd.DataFrame({
    'model': [f'{model_name_save}'],
    'rmse': [rmse],
    'bic': [bic],
    'sabic': [sabic]
}, index=[0]).round(4)
fit_metrics

fit_metrics.to_csv(here(f'fit_metrics/{model_name_save}.csv'))