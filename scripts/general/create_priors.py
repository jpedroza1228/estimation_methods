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

# pdina_mod, pdina_fit = joblib.load("data/joblib_models/prior_DINA_no_testlet_5attr_uniform.joblib")
# dina_mod, dina_fit = joblib.load("data/joblib_models/DINA_no_testlet_5attr_uniform.joblib")

# fitdf = dina_fit.draws_pd()

# nudf = fitdf.filter(regex = r'^nu')
# lognudf = fitdf.filter(regex = r'^log_nu')

# nudf.columns.tolist()

# nudf.mean()

# np.mean(np.random.beta(1.5, 23.5))

# pdina_mod, pdina_fit = joblib.load("data/joblib_models/prior_DINA_no_testlet_5attr_informative.joblib")

dina_mod, dina_fit = joblib.load("data/joblib_models/DINA_no_testlet_5attr_informative.joblib")

fitdf = dina_fit.draws_pd()

nudf = fitdf.filter(regex = r'^nu')
lognudf = fitdf.filter(regex = r'^log_nu')
nudf.columns.tolist()

nudf.mean()
nudf.var()
np.exp(lognudf.mean())
np.exp(lognudf.var())

np.mean(np.random.beta(1.5, 23.5))

slipdf = fitdf.filter(regex = r'slip')
slipdf.mean()
slipdf.var()

guessdf = fitdf.filter(regex = r'guess')
guessdf.mean()
guessdf.var()


# DINO model

# data/joblib_models/prior_DINO_no_testlet_5attr_informative.joblib
# data/joblib_models/DINO_no_testlet_5attr_informative.joblib

# data/joblib_models/DINO_no_testlet_5attr_uniform.joblib
# data/joblib_models/prior_DINO_no_testlet_5attr_uniform.joblib