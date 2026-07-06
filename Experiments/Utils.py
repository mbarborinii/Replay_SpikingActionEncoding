import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from Utils_Functions.DynamicsNet import DynamicsNet, PhysicsConsistentLoss
from Utils_Functions.ExperimentLogger import ExperimentLogger
from Utils_Functions.Module import Module
from Utils_Functions.Executor_Modular import make_executor_report1
from Utils_Functions import Neuron_Models as neuron_models


def _npz_field_to_epoch_list(field: np.ndarray) -> list:
    """
    Convert one npz field (as loaded by np.load) into a list of per-epoch
    numpy arrays, regardless of whether it was saved as an object array
    (each element already an independent array/list) or as one dense
    numeric array (one leading "epoch" axis). Never returns nested plain
    Python lists/floats for the per-epoch entries -- only ndarrays (or,
    for spikes_rasters, lists of ndarrays, since each epoch there is
    itself a list of per-timestep arrays of possibly differing shape).
    """
    if field.dtype == object:
        return list(field.tolist())
    return [np.asarray(field[i]) for i in range(field.shape[0])]


def log_params(params_path, rasters_path: Optional[str] = None) -> Dict:
    """
    Load a params dict from .pt or .npz, attaching spikes_rasters from a companion
    file if the main file doesn't already carry them (npz params and rasters are
    saved as two separate files by ExperimentLogger.save_params()).

    Returns a dict guaranteed to have keys: 'output', 'target', 'spikes_rasters',
    where each is a list (one entry per saved epoch) of numpy arrays -- never a
    deeply-nested plain Python list. This matters because np.savez may have
    stored 'output'/'target' as a single dense array (one ndarray with shape
    (n_epochs, B, T, F)) rather than an object array of per-epoch arrays, in
    which case a naive .tolist() recursively converts every number to plain
    Python floats and the per-epoch entries lose .shape entirely.
    """
    if params_path.endswith(".pt"):
        params = torch.load(params_path, weights_only=False)
    else:
        f = np.load(params_path, allow_pickle=True)
        params = {k: _npz_field_to_epoch_list(f[k]) for k in f.files}

    if "spikes_rasters" not in params or not params.get("spikes_rasters"):
        if rasters_path is not None and os.path.isfile(rasters_path):
            r = np.load(rasters_path, allow_pickle=True)
            params["spikes_rasters"] = _npz_field_to_epoch_list(r["spikes_rasters"])
        else:
            params.setdefault("spikes_rasters", [])

    return params


def run_experiment_helper(experiment_name):
    
    # the dataset is already saved in 
    results_name = str("Results/" + experiment_name)
    logger = ExperimentLogger(results_name)
    params = logger.load_model_full(model_name = str("Parameters/" + experiment_name))
    
    # assumes standard network structure
    model = Module(name = experiment_name, params=params, layer1 = neuron_models.LIF, layer2 = neuron_models.SE_adRLIF, layer3=neuron_models.LI)
    
    executor = make_executor_report1(logger = logger, module = model ,interleave = False if "INHEXC" in experiment_name else True)
    executor.test(plot=False)
    
    
    # Test and train sets?
    # Add plotting for one of the example trajectory + replative rasterplot? 
    
    return 
    
