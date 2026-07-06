
import warnings
import os
import sys
import subprocess
import argparse
import numpy as np
import matplotlib.pyplot as plt 
from scipy.fftpack import dst
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Utils_Functions.Utils import plot, delete


import time
import inspect
import copy


def filter_kwargs_for_constructor(cls, kwargs: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    valid_keys = set(sig.parameters) - {"self"}

    return {k: v for k, v in kwargs.items() if k in valid_keys}

def deep_update(dst: dict, src: dict, *, strict: bool = False):
    for k, v in src.items():
        if strict and k not in dst:
            raise KeyError(f"Unknown param key: {k}")
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v, strict=strict)
        else:
            dst[k] = v




class Module():

    # consider making a module.state_dict() for saving intermediate steps?
    def __init__(self, name, params, layer1, layer2=None, layer3=None, opt_dicts = None):
        
        self.name = name
        self.params = params
        self.gp = params["general_params"]
        batch_size = self.gp["batch_size"]
        nb_batches = self.gp["nb_batches"]
        n_samples = nb_batches * batch_size
        self.device = self.gp["device"]
        self.dtype = self.gp["dtype"]
        
        if opt_dicts is not None:
            self.optimizer_dictionaries = opt_dicts
            
        this_is_not_ModelComplete_check = True # TODO try to figure how to have model complete here
        
        try:
            self.tp = params["training_params"]
        except KeyError:
            # warnings.warn("Warning: No training parameters")
            pass
        try:
            self.epp = params["encoder_population_params"]
        except KeyError:
            warnings.warn("Warning: No encoding layer")
            pass
        try:
            self.rpp = params["recurrent_population_params"]
        except KeyError:
            # warnings.warn("Warning: No recurrent layer")
            pass
        try:
            self.dpp = params["decoder_population_params"]
        except KeyError:
            # warnings.warn("Warning: No decoding layer")
            pass

        self.train_dataloader = DataLoader(self.gp["train_dataset"], batch_size=batch_size, shuffle=False)
        self.test_dataloader = DataLoader(self.gp["test_dataset"], batch_size=batch_size, shuffle=False) if self.gp.get("test_dataset") is not None else None

        self.train_data = self.gp["train_dataset"] if hasattr(self.gp["train_dataset"], "data") else None
        self.test_data = self.gp["test_dataset"] if hasattr(self.gp["test_dataset"], "data") else None

        self.lr = self.tp.get("lr", 0.001) if self.tp is not None else 0.001
        
        if this_is_not_ModelComplete_check:
            
            if layer2 is None:
                #assumes we just have the encoding layer, no training
                merged_1 = {**self.gp, **self.epp}
                kwargs_1 = filter_kwargs_for_constructor(layer1, merged_1)
                self.layer1 = layer1(**kwargs_1)
            else: 
                if layer3 is None:
                    # assumes we have  recurrent and decoding 
                    merged_1 = {**self.gp, **self.rpp}
                    kwargs_1 = filter_kwargs_for_constructor(layer1, merged_1)
                    self.layer1 = layer1(**kwargs_1)
                    
                    merged_2 = {**self.gp, **self.dpp}
                    kwargs_2 = filter_kwargs_for_constructor(layer2, merged_2)
                    self.layer2 = layer2(**kwargs_2)        
                else:
                    # assumes we have 3 separate layers: encoding recurrent decoding
                    merged_1 = {**self.gp, **self.epp}
                    kwargs_1 = filter_kwargs_for_constructor(layer1, merged_1)
                    self.layer1 = layer1(**kwargs_1)
                    
                    merged_2 = {**self.gp, **self.rpp}
                    kwargs_2 = filter_kwargs_for_constructor(layer2, merged_2)
                    self.layer2 = layer2(**kwargs_2)
                    
                    merged_3 = {**self.gp, **self.dpp}
                    kwargs_3 = filter_kwargs_for_constructor(layer3, merged_3)
                    self.layer3 = layer3(**kwargs_3)
        else:
            # assumes we have the ModelComplete aka one layer has everything
            # TODO implement modelcomplete con encoding layer
            merged_1 = {**self.gp, **self.tp, **self.epp, **self.rpp, **self.dpp}
            kwargs_1 = filter_kwargs_for_constructor(layer1, merged_1)
            self.layer1 = layer1(**kwargs_1)


        self.layer1_state_dict = self.layer1.state_dict()
        self.layer2_state_dict = self.layer2.state_dict() if hasattr(self, 'layer2') else None
        self.layer3_state_dict = self.layer3.state_dict() if hasattr(self, 'layer3') else None
        
        opt_list = []
        opt_list_names = [] 
  
        for name, value in self.layer1_state_dict.items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                opt_list.append(value)
                opt_list_names.append(f"layer_1_{name}")
        if self.layer2_state_dict is not None:
            for name, value in self.layer2_state_dict.items():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    opt_list.append(value)
                    opt_list_names.append(f"layer_2_{name}")
        if self.layer3_state_dict is not None:
            for name, value in self.layer3_state_dict.items():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    opt_list.append(value)
                    opt_list_names.append(f"layer_3_{name}")
                    
                    
        self.opt_list_names = opt_list_names       
        self.opt_list = opt_list 
        self.optimizer = self.tp.get("optimizer")
            
            
        if self.optimizer == "Adamax":
            self.optimizer_name = "Adamax"
            self.optimizer = torch.optim.Adamax(self.opt_list, lr=self.tp["lr"], betas=(0.9, 0.9))
        # possibility to add different optimizers here if needed
        if self.optimizer == "Adam":
            self.optimizer_name = "Adam"
            self.optimizer = torch.optim.Adam(self.opt_list, lr=self.tp["lr"], betas=(0.9, 0.999))
        
        self.runtime_percentage_teaching = self.tp.get("runtime_percentage_teaching", 1.0)
        self.dynamics_net = self.tp["dynamics_net"] if self.tp.get("dynamics_net") is not None else None
        self.loss_fn = self.tp["loss_fn"] 



    def state_dict(self):
        layer1 = str(self.layer1.name)
        layer2 = str(self.layer2.name) if hasattr(self, "layer2") else None
        layer3 = str(self.layer3.name) if hasattr(self, "layer3") else None

        return {
                layer1: self.layer1.state_dict(),
                layer2: self.layer2.state_dict() if hasattr(self, "layer2") else None,
                layer3: self.layer3.state_dict() if hasattr(self, "layer3") else None
        }
        
    def state_dict_variant(self):
        layer1 = str(self.layer1.name)
        layer2 = str(self.layer2.name) if hasattr(self, "layer2") else None
        layer3 = str(self.layer3.name) if hasattr(self, "layer3") else None
        
        if layer1 == "GaussPop":
            layer1 = "encoder_population_params"
        if layer2 == "SE_adRLIF":
            layer2 = "recurrent_population_params"
        if layer3 == "LI":
            layer3 = "decoder_population_params"

        return {
                layer1: self.layer1.state_dict(),
                layer2: self.layer2.state_dict() if hasattr(self, "layer2") else None,
                layer3: self.layer3.state_dict() if hasattr(self, "layer3") else None
        }

    def load_dict(self):
        # To load all the Module variables
        return self.__dict__


    
    def detach_state(self, obj):
        """Recursively convert Parameters/Tensors into detached CPU tensors."""
        if isinstance(obj, torch.nn.Parameter):
            return obj.detach().cpu()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, dict):
            return {k: self.detach_state(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.detach_state(v) for v in obj)
        else:
            # int, float, str, bool, None, etc.
            return obj
    
    
    
    def set_optimizer(self, dictionary):
        
        opt_list_temp = [] 
        opt_list_names_temp = []
        
        for name,value in dictionary.items():
            
            if isinstance(value, float):
                # assumes that, if we have a float, it is the learning rate
                lr = value
            
            if isinstance(value, list):
                for object in value:
                    # Since I assume that the name in the original dictionary is the same as the instance in the original dictionaries

                    # bag, path = find_nested_value(self.state_dict_variant(), name, object )
                    opt_list_temp.append(self.state_dict_variant()[name][object])
                    opt_list_names_temp.append(f"{name}_{object}")
                    assert(self.state_dict_variant()[name][object].requires_grad == True)

        self.opt_list_temp = opt_list_temp
        self.opt_list_names_temp = opt_list_names_temp
        if self.optimizer_name == "Adamax":
            self.optimizer = torch.optim.Adamax(self.opt_list_temp, lr=lr, betas=(0.9, 0.9))
        # possibility to add different optimizers here if needed
        if self.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.opt_list_temp, lr=lr, betas=(0.9, 0.999))
        
        
        
    def reset_module(self, new_params: dict, *, strict: bool = False):
        """
        Reset module using updated params.
        strict=False: allow partial update
        strict=True: require all keys to exist
        """
        # 1) merge params (deep merge is best if nested dicts)
        # updated_params = copy.deepcopy(self.params)
        deep_update(self.params, new_params, strict=strict)

        # 2) remember which layer classes were used
        layer1_cls = self.layer1.__class__
        layer2_cls = self.layer2.__class__ if hasattr(self, "layer2") else None
        layer3_cls = self.layer3.__class__ if hasattr(self, "layer3") else None

        # 3) rebuild a fresh module instance
        new_obj = self.__class__(
        name=self.name,
        params=self.params,
        layer1=layer1_cls,
        layer2=layer2_cls,
        layer3=layer3_cls,
        )

        # 4) replace state
        self.__dict__.clear()
        self.__dict__.update(new_obj.__dict__)
                
        
    def change_optimizer_list(self, learning_rate=None):
        
        opt_list = []
        for _, value in self.layer1_state_dict.items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                opt_list.append(value)
        for _, value in self.layer2_state_dict.items():
            if isinstance(value, torch.Tensor) and value.requires_grad:
                opt_list.append(value)
        
        if learning_rate is not None:
            self.tp["lr"] = learning_rate
        
        self.optimizer = self.tp.get("optimizer", "Adamax")
        if self.optimizer == "Adamax":
            self.optimizer = torch.optim.Adamax(opt_list, lr=self.tp["lr"], betas=(0.9, 0.9))
        # possibility to add different optimizers here if needed
        
        return self.optimizer