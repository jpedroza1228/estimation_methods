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
stan_model_directory = 'stan_models/rdina_working.stan'
stan_prior_directory = 'stan_models/rdina_working_prior.stan'
diagnostics_directory = 'working_diagnostics.csv'
save_model_joblib_directory = 'working_model.joblib'
save_prior_joblib_directory = 'working_prior.joblib'
item_ppp_directory = 'working_ppp.csv'
resp_proficieny_raw_list_directory = 'working_prof_raw.csv'
resp_proficiency_list_directory = 'working_prof_list.csv'

threshold = .8
model_type = 'RDINA'

attr1 = 'expect_fun'
attr2 = 'def_var'
attr3 = 'prop_expect'
attr4 = 'expect_famous_dist'
attr5 = 'prop_var'

df = pd.read_csv(here(f'{data_directory}')).drop(columns = 'Unnamed: 0')
q = pd.read_csv(here(f'{q_directory}')).clean_names(case_type = 'snake').drop(columns = 'unnamed_0')
q['item'] = ['item' + str(i) for i in q['item']]

# if no testlet, run this code
q = q.drop(columns = 'question')

print(df.head())
print(q.head())

df = df.drop(columns = 'item35')
df.head()

q = q.loc[q['item'] != 'item35']
q.head()


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
      xi[i,c] = 1 - np.prod((1 - alpha.drop(columns = ['index', 'class_label']).to_numpy()[c, :]) ** q.drop(columns = 'item').to_numpy()[i, :])

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


file = os.path.join(here(f'{stan_model_directory}'))
model = CmdStanModel(stan_file = file,
                     cpp_options = {'STAN_THREADS': 'TRUE'})

np.random.seed(7102026)
fit = model.sample(data = stan_dict,
                     show_console = True,
                     chains = 4,
                     parallel_chains = 4,
                     adapt_delta = .99,
                     iter_warmup = 2000,
                     iter_sampling = 2000)


diagnose = pd.DataFrame(fit.summary())
diagnose.sort_values('R_hat', ascending = False)
# diagnose.to_csv(here(f'{diagnostics_directory}'))
                   
joblib.dump([model, fit],
            here(f'{save_model_joblib_directory}'),
            compress = 3)

prior_file = os.path.join(here(f'{stan_prior_directory}'))
pmodel = CmdStanModel(stan_file = prior_file,
                       cpp_options = {'STAN_THREADS': 'TRUE'})

np.random.seed(7102026)
pfit = pmodel.sample(data = stan_dict,
                     show_console = True,
                     chains = 4,
                     parallel_chains = 4,
                     adapt_delta = .99,
                     iter_warmup = 2000,
                     iter_sampling = 2000)

# joblib.dump([pmodel, pfit],
#             here(f'{save_prior_joblib_directory}'),
#             compress = 3)

def q_lower(x):
    return x.quantile(.025)
  
def q_upper(x):
    return x.quantile(.975)
  

idata = azb.from_cmdstanpy(
    posterior = fit,
    prior = pfit,
    posterior_predictive = ['y_rep'],
    prior_predictive = ['y_rep'],
    observed_data = {'y_rep': stan_dict['Y']},
    log_likelihood = 'log_lik'
    )

fitdf = fit.draws_pd()

azp.plot_trace(idata, var_names = ['noise'])
azp.plot_trace(idata, var_names = ['nu'])

azp.plot_prior_posterior(idata, var_names = ['noise'], kind = 'kde')
azp.plot_prior_posterior(idata, var_names = ['nu'], kind = 'kde')


azp.plot_ppc_dist(idata, kind = 'kde', group = 'posterior_predictive')

azp.plot_ppc_tstat(idata, t_stat = 'mean')
azp.plot_ppc_tstat(idata, t_stat = 'std')


noise = fitdf.filter(regex = 'noise').reset_index()
noise = noise.rename(columns = {'index': 'draw'})

noiselong = noise.melt(id_vars = 'draw')
noiselong[['type', 'item']] = noiselong['variable'].str.split('[', expand = True)
noiselong['item'] = noiselong['item'].str.replace(']', '')
noiselong = noiselong[['draw', 'type', 'item', 'value']]
noiselong[['draw', 'item']] = noiselong[['draw', 'item']].astype(int)

noiseavg = noiselong.groupby('item')

noiseavg = pd.DataFrame({
  'mean': noiseavg['value'].mean(),
  'std': noiseavg['value'].std(),
  'q_lower': q_lower(noiseavg['value']),
  'q_upper': q_upper(noiseavg['value'])
}).reset_index()

# noiseavg

# pi matrix (classes getting items correct)
pidf = fitdf.filter(regex = 'pi').reset_index()
pidf = pidf.rename(columns = {'index': 'draw'})
pilong = pidf.melt(id_vars = 'draw')
pilong['variable'] = pilong['variable'].str.replace('pi[', '')
pilong['variable'] = pilong['variable'].str.replace(']', '')
pilong[['item', 'latclass']] = pilong['variable'].str.split(',', expand = True)
pilong = pilong[['draw', 'item', 'latclass', 'value']]
pilong[['draw', 'item', 'latclass']] = pilong[['draw', 'item', 'latclass']].astype(int)

piavg = pilong.groupby(['item', 'latclass'])['value'].agg(['mean', 'std', q_lower, q_upper]).reset_index()

piavg = piavg.merge(alpha, 'left', left_on = 'latclass', right_on = 'index')

# piavg

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

# t-values
y_describe = y_item.filter(regex = 'item').agg(['mean', 'std']).reset_index()
y_describe.drop(columns = 'index').transpose().round(2)

stu_n = y_item.shape[0]

t_stats_dict = {}

# Loop from 1 to 7
for i in range(1, stan_dict['I']):
    item_name = f"item{i}"
    
    # Extract mean and std for the specific item
    # We use .values[0] to get the scalar number out of the filtered dataframe
    avg = y_describe.loc[y_describe['index'] == 'mean', item_name].values[0]
    std = y_describe.loc[y_describe['index'] == 'std', item_name].values[0]
    
    # Calculate the observed t-value
    # Formula: T = avg / (std / sqrt(n))
    t_val = avg / ((std + 1e-10) / np.sqrt(stu_n))
    
    # Store it in our dictionary
    t_stats_dict[item_name] = t_val

# Convert the dictionary into a final Pandas Series
obs_t_series = pd.Series(t_stats_dict)

obs_t_series = obs_t_series.reset_index()
obs_t_series = obs_t_series.rename(columns = {'index': 'item',
                                              0: 'observed_t'})
obs_t_series['item'] = obs_t_series['item'].str.replace('item', '')
obs_t_series['item'] = obs_t_series['item'].astype(float)

y_long_avg = ydcm_long.groupby(['item', 'draw'])['value'].agg(['mean', 'std']).reset_index()
y_long_avg['n'] = stu_n

y_long_avg['t_draw'] = y_long_avg['mean']/((y_long_avg['std'] + 1e-10)/np.sqrt(y_long_avg['n']))

y_long_avg = y_long_avg.merge(obs_t_series, 'inner', 'item')

# attribute calculations
attrdf = fitdf.filter(regex = '^prob_resp_attr').reset_index()
attrdf = attrdf.rename(columns = {'index': 'draw'})
attrlong = attrdf.melt(id_vars = 'draw')

attrlong['variable'] = attrlong['variable'].str.replace('prob_resp_attr[', '')
attrlong['variable'] = attrlong['variable'].str.replace(']', '')
attrlong[['stu', 'attr']] = attrlong['variable'].str.split(',', expand = True)
attrlong[['draw', 'stu', 'attr']] = attrlong[['draw', 'stu', 'attr']].astype(int)
attrlong = attrlong[['draw', 'stu', 'attr', 'value']]

attravg = attrlong.groupby(['stu', 'attr'])['value'].agg(['mean', 'std', q_lower, q_upper]).reset_index()

# mastery/proficiency thresholds
attravg['mastery'] = np.where(attravg['mean'] > threshold, 1, 0)

attr_w = attravg.pivot(index = 'stu', columns = 'attr', values = 'mastery').reset_index()

# reliability
attravg['acc_comp'] = attravg['mean'].apply(lambda p: max(p, 1 - p))
attravg['cons_comp'] = attravg['mean'].apply(lambda p: p**2 + (1 - p)**2)

# latent classes
attr_class = fitdf.filter(regex = '^prob_resp_class').reset_index()
attr_class = attr_class.rename(columns = {'index': 'draw'})
class_long = attr_class.melt(id_vars = 'draw')

class_long['variable'] = class_long['variable'].str.replace('prob_resp_class[', '')
class_long['variable'] = class_long['variable'].str.replace(']', '')
class_long[['stu', 'latclass']] = class_long['variable'].str.split(',', expand = True)
class_long[['draw', 'stu', 'latclass']] = class_long[['draw', 'stu', 'latclass']].astype(int)
class_long = class_long[['draw', 'stu', 'latclass', 'value']]

class_max = class_long.loc[class_long['value'] == class_long.groupby('stu')['value'].transform('max')]
class_max['high_prob_class'] = class_max['latclass']

class_wmax = class_long.merge(class_max, 'inner', ['latclass', 'stu'])
class_wmax = class_wmax.drop(columns = ['draw_y', 'latclass', 'value_y'])
class_wmax = class_wmax.rename(columns = {'draw_x': 'draw',
                             'value_x': 'value',
                             'high_prob_class': 'class_member'})

class_avg = class_wmax.groupby(['stu', 'class_member'])['value'].agg(['mean', 'std', q_lower, q_upper]).reset_index()
class_avg = class_avg.merge(alpha, 'inner', left_on = 'class_member', right_on = 'index')

class_prop = class_long.groupby('latclass')['value'].agg(['mean', 'std', q_lower, q_upper]).reset_index()
class_prop = class_prop.merge(alpha, 'inner', left_on = 'latclass', right_on = 'index')


(
  pn.ggplot(ydcm_wide_count.loc[ydcm_wide_count['type'] != 'draw_counts'],
            pn.aes('total',
                   'count'))
  + pn.geom_errorbar(pn.aes(ymin = 'lower',
                            ymax = 'upper'),
                     alpha = .5,
                     linetype = 'dashed')
  + pn.geom_point(pn.aes(color = 'type'), alpha = .8)
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.scale_x_continuous(limits = [1, piavg['item'].unique().max()],
                          breaks = piavg['item'].unique())
  + pn.labs(title = 'Comparisons between Actual Counts of Respondents\nGetting Items/Questions Correct',
  subtitle = 'And Posterior Average Correct')
)

print('Average correct per item:', y_long_avg['t_draw'].describe())

t_compare_list = [np.mean(y_long_avg.loc[(y_long_avg['item'] == i), 't_draw'] > y_long_avg.loc[(y_long_avg['item'] == i), 'observed_t']) for i in np.arange(1, stan_dict['I'])]

gt(pd.DataFrame({'item': np.arange(1, stan_dict['I']),
                 't_prop_over': t_compare_list}).round(2)).tab_header(title = 'Item Level PPP')

(
  pn.ggplot(class_avg.loc[class_avg['mean'] < .5],
    pn.aes('stu', 'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
    linetype = 'dashed',
    alpha = .7)
  + pn.geom_point(pn.aes(color = 'factor(class_label)'),
                  alpha = .7)
  + pn.coord_flip()
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.labs(title = 'Probability of Membership in Latent Classes',
            subtitle = 'For Each Respondent')
)

(
  pn.ggplot(class_avg.loc[class_avg['mean'].between(.5, .75)],
    pn.aes('stu', 'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
    linetype = 'dashed',
    alpha = .7)
  + pn.geom_point(pn.aes(color = 'factor(class_label)'),
                  alpha = .7)
  + pn.coord_flip()
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.labs(title = 'Probability of Membership in Latent Classes',
            subtitle = 'For Each Respondent')
)


(
  pn.ggplot(class_avg.loc[(class_avg['mean'] < .95) & (class_avg['mean'] > .75)],
    pn.aes('stu', 'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
    linetype = 'dashed',
    alpha = .7)
  + pn.geom_point(pn.aes(color = 'factor(class_label)'),
                  alpha = .7)
  + pn.coord_flip()
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.labs(title = 'Probability of Membership in Latent Classes',
            subtitle = 'For Each Respondent')
)

(
  pn.ggplot(class_avg.loc[class_avg['mean'] > .95],
    pn.aes('stu', 'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
    linetype = 'dashed',
    alpha = .7)
  + pn.geom_point(pn.aes(color = 'factor(class_label)'),
                  alpha = .7)
  + pn.coord_flip()
  + pn.scale_color_brewer('qual', 'Dark2')
  + pn.labs(title = 'Probability of Membership in Latent Classes',
            subtitle = 'For Each Respondent')
)


(
  pn.ggplot(class_prop,
            pn.aes('factor(class_label)',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     linetype = 'dashed',
                     alpha = .7)
  + pn.geom_point(alpha = .7,
                  color = 'seagreen')
  + pn.coord_flip()
  + pn.labs(title = 'Probability of Latent Class Proportions')
)



(
  pn.ggplot(attravg,
            pn.aes('stu',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     color = 'seagreen',
                     alpha = .1)
  + pn.geom_point(alpha = .7,
                  color = 'seagreen')
  + pn.geom_hline(yintercept = threshold,
                  color = 'black',
                  linetype = 'dashed')
  + pn.facet_wrap('attr')
  + pn.theme(legend_position = 'none',
             axis_text_x = pn.element_blank())
)


attr_wide = pd.concat([df['anon_id'], attr_w], axis = 1)
attr_wide = attr_wide.drop(columns = 'stu')
attr_wide.columns = ['anon_id', attr1, attr2, attr3, attr4, attr5]

attr_wide


(
  pn.ggplot(noiseavg,
    pn.aes('factor(item)', 'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
  linetype = 'dashed',
  alpha = .7)
  + pn.geom_point(color = 'seagreen', alpha = .7)
  + pn.coord_flip()
  + pn.scale_y_continuous(limits = [0, 1],
                          breaks = np.arange(0, 1.1, .1))
  + pn.labs(title = 'Probability of Noise',
            x = 'Item',
            y = 'Probability',
            caption = 'Guess = Guessed and got answer correct\nSlip = Should know the answer, but got the item wrong')  
  + pn.theme(legend_position = 'none')
)


(
  pn.ggplot(piavg,
            pn.aes('item',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     linetype = 'dashed',
                     color = 'gray',
                     alpha = .7)
  + pn.geom_point(alpha = .7,
                  color = 'seagreen')
  + pn.geom_hline(yintercept = .5,
  color = 'black',
  linetype = 'dashed')
  + pn.scale_x_continuous(limits = [1, piavg['item'].unique().max()],
                          breaks = piavg['item'].unique())
  + pn.coord_flip()
  + pn.facet_wrap('class_label')
  + pn.labs(title = 'Probability of Getting Items Correct',
  subtitle = 'By Latent Class')
  + pn.theme(legend_position = 'none')
)


(
  pn.ggplot(piavg.loc[piavg['class_label'] == '11111'],
            pn.aes('item',
                   'mean'))
  + pn.geom_errorbar(pn.aes(ymin = 'q_lower', ymax = 'q_upper'),
                     linetype = 'dashed',
                     color = 'gray',
                     alpha = .7)
  + pn.geom_point(alpha = .7,
                  color = 'seagreen')
  + pn.geom_hline(yintercept = .5,
  color = 'black',
  linetype = 'dashed')
  + pn.scale_x_continuous(limits = [1, piavg['item'].unique().max()],
                          breaks = piavg['item'].unique())
  + pn.coord_flip()
  + pn.facet_wrap('class_label')
  + pn.labs(title = 'Probability of Getting Items Correct',
  subtitle = 'By Latent Class')
  + pn.theme(legend_position = 'none')
)

reliability = attravg.groupby('attr').agg(
    accuracy = ('acc_comp', 'mean'),
    consistency = ('cons_comp', 'mean')
).reset_index()

gt(reliability.round(3)).tab_header(title = 'Accuracy & Consistency')


np.mean(np.random.beta(5, 20))