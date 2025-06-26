
import os
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# This is the normalization script adapted from FTCP work.
# It works for FTCP, IRCR.

def minmax(pcr): 
    dim0, dim1, dim2 = pcr.shape
    scaler = MinMaxScaler()
    pcr_ = np.transpose(pcr, (1, 0, 2))
    pcr_ = pcr_.reshape(dim1, dim0*dim2)
    pcr_ = scaler.fit_transform(pcr_.T)
    pcr_ = pcr_.T
    pcr_ = pcr_.reshape(dim1, dim0, dim2)
    pcr_normed = np.transpose(pcr_, (1, 0, 2))
    return pcr_normed, scaler

def inv_minmax(pcr_normed, scaler):
    dim0, dim1, dim2 = pcr_normed.shape
    pcr_ = np.transpose(pcr_normed, (1, 0, 2))
    pcr_ = pcr_.reshape(dim1, dim0*dim2)
    pcr_ = scaler.inverse_transform(pcr_.T)
    pcr_ = pcr_.T
    pcr_ = pcr_.reshape(dim1, dim0, dim2)
    pcr = np.transpose(pcr_, (1, 0, 2))
    return pcr

def load_saved_diffusion_scaler(path=None):
    if path is None:
        here = os.path.dirname(__file__)
        path = os.path.join(here, "..", "assets", "ircr_diffusion_scaler.pkl")
        path = os.path.abspath(path)
    return joblib.load(path)


