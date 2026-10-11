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

data_directory = 'data/stat_134_2025_quiz5.csv'
q_directory = 'q_matrix/quiz5_q_5attr.csv'
stan_model_directory = 'stan_models/no_testlet/non_informative_priors/rdino_rdina_model.stan'
stan_prior_directory = 'stan_models/no_testlet/non_informative_priors/rdino_rdina_prior.stan'
diagnostics_directory = 'data/diagnostics/rdina_no_testlet_attr5_model_diagnostics.csv'
save_model_joblib_directory = 'data/joblib_models/no_inform_prior_no_testlet_5attr_rdina_model.joblib'
save_prior_joblib_directory = 'data/joblib_models/no_inform_prior_no_testlet_5attr_rdina_prior.joblib'
item_ppp_directory = 'data/item_ppp/no_inform_no_testlet_attr5_rdina.csv'
resp_proficieny_raw_list_directory = 'data/mastery_lists/resp_prof_raw_no_inform_no_testlet_attr5_rdina.csv'
resp_proficiency_list_directory = 'data/mastery_lists/resp_prof_list_no_inform_no_testlet_attr5_rdina.csv'

threshold = .8
model_type = 'RDINA'

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

if model_type == 'RDINO':
  # DINO model
  for i in range(q.drop(columns = 'item').shape[0]):
    for c in range(alpha.drop(columns = ['index', 'class_label']).shape[0]):
      dino_eta[i,c] = 1 - np.prod((1 - alpha.drop(columns = ['index', 'class_label']).to_numpy()[c, :]) ** q.drop(columns = 'item').to_numpy()[i, :])

elif model_type == 'RDINA':
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

# Load the preserved Python object (e.g., a machine learning model or array)
model, fit = joblib.load(here('data/joblib_models/no_inform_prior_no_testlet_5attr_rdina_model.joblib'))

pmodel, pfit = joblib.load(here('data/joblib_models/no_inform_prior_no_testlet_5attr_rdina_prior.joblib'))

idata = azb.from_cmdstanpy(
    posterior = fit,
    prior = pfit,
    posterior_predictive = ['y_rep'],
    prior_predictive = ['y_rep'],
    observed_data = {'y_rep': stan_dict['Y']},
    log_likelihood = 'log_lik'
    )

fitdf = fit.draws_pd()

# replicated data
ydcm = fitdf.filter(regex = '^y_rep')

# calculations for odds ratios/conditional probabilities
ydcm_long = ydcm.melt()

ydcm_long['variable'] = ydcm_long['variable'].str.replace('y_rep[', '')
ydcm_long['variable'] = ydcm_long['variable'].str.replace(']', '')
ydcm_long[['stu', 'item']] = ydcm_long['variable'].str.split(',', expand = True)

ydcm_long = ydcm_long[['stu', 'item', 'value']]
ydcm_long[['stu', 'item']] = ydcm_long[['stu', 'item']].astype(int)
ydcm_long['draw'] = ydcm_long.groupby(['stu', 'item']).cumcount()

ydcm_wide = ydcm_long.pivot(index = ['stu', 'draw'], columns = 'item', values = 'value')
ydcm_wide = ydcm_wide.reset_index()

y_item_columns = [f'item{i}' for i in np.arange(stan_dict['I']) + 1]
ydcm_wide.columns = ['stu', 'draw'] + y_item_columns

ydcm_wide['total'] = ydcm_wide.filter(regex = 'item').sum(axis = 1)
ydcm_wide_count = ydcm_wide.groupby('draw')['total'].value_counts().reset_index()

ydcm_scores = ydcm_wide_count.groupby('total')['count'].agg(
    count = 'mean',
    lower = q_lower,
    upper = q_upper
).reset_index()

ydcm_wide_count['type'] = 'draw_counts'
ydcm_scores['type'] = 'avg_counts'

ydcm_wide_count['count'] = ydcm_wide_count['count'].astype(float)
ydcm_wide_count = ydcm_wide_count.merge(ydcm_scores, 'outer')

y_item = pd.DataFrame(stan_dict['Y'])
y_item.columns = y_item_columns

y_item['total'] = y_item.sum(axis = 1)
y_item_count = y_item['total'].value_counts().reset_index()
y_item_count['type'] = 'actual_counts'
y_item_count['count'] = y_item_count['count'].astype(float)

ydcm_wide_count = ydcm_wide_count.merge(y_item_count, 'outer')

# 
ydcm_wide.head()
y_item = y_item.reset_index()
y_item = y_item.rename(columns = {'index': 'stu'})
y_item['stu'] = y_item['stu'].astype(int) + 1

ydcm_wide_avg = ydcm_wide.groupby('stu')[ydcm_wide.filter(regex = 'item').columns.tolist()].agg(np.mean).reset_index()

ydcm_wide_avg = ydcm_wide_avg.drop(columns = 'stu')

y_compare = y_item.drop(columns = ['stu', 'total'])


from sklearn.metrics import mean_squared_error

# calculate RMSE
rmse = np.sqrt(mean_squared_error(y_compare.values.flatten(), ydcm_wide_avg.values.flatten()))

# 2. Posterior mean of log likelihood
log_lik = fitdf['lp__'].mean()

# 3. Define sample size (num_stu) and estimated parameters (num_param)
# num_param = guessing (I) + slipping (I) + latent class proportions (C - 1)
num_stu = y_item.shape[0]
num_item = y_item.drop(columns = ['stu', 'total']).shape[1]
num_lat_class = alpha.shape[0]
num_param = (2 * num_item) + (num_lat_class - 1)

# 4. Compute Information Criteria
bic = -2 * log_lik + num_param * np.log(num_stu)
sabic = -2 * log_lik + num_param * np.log((num_stu + 2) / 24)

# 5. Construct the metrics DataFrame
fit_metrics = pd.DataFrame({
    'model': ['rdina_no_testlet_non_inform_prior_5attr_mcmc'],
    'rmse': [rmse],
    'bic': [bic],
    'sabic': [sabic]
}, index=[0]).round(4)

# fit_metrics.to_csv(here('data/fit_metrics/rdina_no_testlet_non_inform_prior_5attr_mcmc.csv'))