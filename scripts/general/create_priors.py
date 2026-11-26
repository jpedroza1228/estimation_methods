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

# DINA model

# dina_mod, dina_fit = joblib.load("data/joblib_models/DINA_no_testlet_5attr_informative.joblib")

# fitdf = dina_fit.draws_pd()

# nudf = fitdf.filter(regex = r'^nu')
# lognudf = fitdf.filter(regex = r'^log_nu')
# nudf.columns.tolist()

# nudf.mean()
# nudf.var()
# np.exp(lognudf.mean())
# np.exp(lognudf.var())


# slipdf = fitdf.filter(regex = r'slip')
# longslip = slipdf.melt()
# longslip[['drop', 'item']] = longslip['variable'].str.split('[', expand = True)
# longslip['item'] = longslip['item'].str.replace(']', '')
# longslip = longslip[['item', 'value']]
# longslip['item'] = longslip['item'].astype(int)
# longslip['question'] = np.select([
#     longslip['item'].isin([1, 2, 3]),
#     longslip['item'].isin([4, 5, 6, 7, 8, 9, 10]),
#     longslip['item'].isin([11, 12, 13]),
#     longslip['item'].isin([14, 15, 16, 17, 18, 19, 20]),
#     longslip['item'].isin([21, 22, 23]),
#     longslip['item'].isin([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])],
#           [1, 2, 3, 4, 5, 6],
#           default = 0
# )

# longslip.tail()

# longslip.groupby('question').mean()

# guessdf = fitdf.filter(regex = r'guess')
# longguess = guessdf.melt()
# longguess[['drop', 'item']] = longguess['variable'].str.split('[', expand = True)
# longguess['item'] = longguess['item'].str.replace(']', '')
# longguess = longguess[['item', 'value']]
# longguess['item'] = longguess['item'].astype(int)
# longguess['question'] = np.select([
#     longguess['item'].isin([1, 2, 3]),
#     longguess['item'].isin([4, 5, 6, 7, 8, 9, 10]),
#     longguess['item'].isin([11, 12, 13]),
#     longguess['item'].isin([14, 15, 16, 17, 18, 19, 20]),
#     longguess['item'].isin([21, 22, 23]),
#     longguess['item'].isin([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])],
#           [1, 2, 3, 4, 5, 6],
#           default = 0
# )

# longguess.tail()

# longguess.groupby('question').mean()

# DINO model

# dino_mod, dino_fit = joblib.load("data/joblib_models/DINO_no_testlet_5attr_informative.joblib")

# fitdf = dino_fit.draws_pd()

# slipdf = fitdf.filter(regex = r'slip')
# longslip = slipdf.melt()
# longslip[['drop', 'item']] = longslip['variable'].str.split('[', expand = True)
# longslip['item'] = longslip['item'].str.replace(']', '')
# longslip = longslip[['item', 'value']]
# longslip['item'] = longslip['item'].astype(int)
# longslip['question'] = np.select([
#     longslip['item'].isin([1, 2, 3]),
#     longslip['item'].isin([4, 5, 6, 7, 8, 9, 10]),
#     longslip['item'].isin([11, 12, 13]),
#     longslip['item'].isin([14, 15, 16, 17, 18, 19, 20]),
#     longslip['item'].isin([21, 22, 23]),
#     longslip['item'].isin([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])],
#           [1, 2, 3, 4, 5, 6],
#           default = 0
# )

# longslip.tail()

# longslip.groupby('question').mean()

# a = 11
# b = 30 - a
# for i in np.arange(10):
#   print(np.mean(np.random.beta(a, b, 30)))

# guessdf = fitdf.filter(regex = r'guess')
# longguess = guessdf.melt()
# longguess[['drop', 'item']] = longguess['variable'].str.split('[', expand = True)
# longguess['item'] = longguess['item'].str.replace(']', '')
# longguess = longguess[['item', 'value']]
# longguess['item'] = longguess['item'].astype(int)
# longguess['question'] = np.select([
#     longguess['item'].isin([1, 2, 3]),
#     longguess['item'].isin([4, 5, 6, 7, 8, 9, 10]),
#     longguess['item'].isin([11, 12, 13]),
#     longguess['item'].isin([14, 15, 16, 17, 18, 19, 20]),
#     longguess['item'].isin([21, 22, 23]),
#     longguess['item'].isin([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34])],
#           [1, 2, 3, 4, 5, 6],
#           default = 0
# )

# longguess.tail()

# longguess.groupby('question').mean()

# a = 7
# b = 30 - a
# for i in np.arange(10):
#   print(np.mean(np.random.beta(a, b, 30)))

# RDINA model

# rdina_mod, rdina_fit = joblib.load("data/joblib_models/RDINA_no_testlet_5attr_informative.joblib")

# fitdf = rdina_fit.draws_pd()
# noisedf = fitdf.filter(regex = r'noise')

# noisedf.mean()

# a = 8
# b = 30 - a
# for i in np.arange(10):
#   print(np.mean(np.random.beta(a, b, 30)))

# RDINO model

# rdino_mod, rdino_fit = joblib.load("data/joblib_models/RDINO_no_testlet_5attr_informative.joblib")

# fitdf = rdino_fit.draws_pd()
# noisedf = fitdf.filter(regex = r'noise')

# noisedf.mean()

# a = 9
# b = 30 - a
# for i in np.arange(10):
#   print(np.mean(np.random.beta(a, b, 30)))