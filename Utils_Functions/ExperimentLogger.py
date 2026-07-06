import os
import json
import csv
import logging
from datetime import datetime
import torch
import matplotlib.pyplot as plt
from Utils_Functions.Utils import plot
import numpy as np



class ExperimentLogger:
    def __init__(self, base_dir="results", name=None):
        self.exp_dir = os.path.join(base_dir, f"{name}" if name else "generic_experiment")
        self.models_dir = os.path.join(self.exp_dir, "models")
        self.figures_dir = os.path.join(self.exp_dir, "figures")
        self.logs_dir = os.path.join(self.exp_dir, "logs")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize logger
        self.logger = self._setup_logger(name)
        
        self.logger.info(f"Experiment created: {self.exp_dir}")
        print(f"Experiment created: {self.exp_dir}")
    
    def _setup_logger(self, name):
        """
        Setup a logger for the experiment with both file and console handlers.
        
        Args:
            name (str): Name of the experiment (used in log filename)
        
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logger with experiment name
        logger_name = name if name else "generic_experiment"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        log_file = os.path.join(self.logs_dir, "experiment.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler (INFO level and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    # --------------------------
    # Params saving/loading
    # --------------------------
    def save_config(self, config):
        """Save experiment configuration."""
        try:
            torch.save(config, f"{self.exp_dir}/shared_params.pt")
            self.logger.info(f"Configuration saved to {self.exp_dir}/shared_params.pt")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}", exc_info=True)



    # --------------------------
    # Model saving/loading
    # --------------------------
    def save_model(self, model):
        """Save model checkpoint with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        path = os.path.join(self.models_dir, f"{model.name}.pt")
        os.makedirs(self.models_dir, exist_ok=True)
        
        try:
            raw_state = model.state_dict()
            # safe_state = model.detach_state(raw_state)
            
            checkpoint = {"timestamp": timestamp,
                          "model_state_dict": raw_state}

            torch.save(checkpoint, path)
            self.logger.info(f"Model saved to {path}")
            print(f"Time {timestamp}: Model saved to {path}")
            return path
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}", exc_info=True)
            raise

    def load_model(self, model):
        """Load model from checkpoint."""
        try:
            path = os.path.join(self.models_dir, f"{model.name}.pt")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["model_state_dict"])
            
            self.logger.info(f"Model loaded from {path}")
            print(f"Model loaded from {path}")
            return checkpoint
        except FileNotFoundError:
            self.logger.error(f"Model file not found: {path}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise
        
    def load_model_full(self, model_name):
        # Assumes above that the folder name is the same as the .pt file to load 
        model_path = (model_name + f"{model_name.split('Parameters')[-1]}.pt")
        shared_path = (model_name + "/shared_params.pt")
        
        model_dict = torch.load(model_path, weights_only=False)
        shared_dict = torch.load(shared_path, weights_only=False)
        
        # I have to change things ad hoc
        shared_dict["general_params"]["figures_directory"] = str("Results/" + model_name.split('Parameters')[-1] + "/figures")
        shared_dict["main_directory"] = str("Results/" + model_name.split('Parameters')[-1])
        
        shared_dict["encoder_population_params"]["weights"]  = model_dict["model_state_dict"]["LIF"]["ff_weights"]  # Initialize the encoder population params
        shared_dict["encoder_population_params"]["beta"] = model_dict["model_state_dict"]["LIF"]["beta"]
        
        # Not included in this specific run, might be needed in others
        
        # shared_dict["recurrent_population_params"]["beta"] = model_dict["model_state_dict"]["SE_adRLIF"]["beta"]
        # shared_dict["recurrent_population_params"]["gamma"] = model_dict["model_state_dict"]["SE_adRLIF"]["gamma"]
        # shared_dict["recurrent_population_params"]["a"] = model_dict["model_state_dict"]["SE_adRLIF"]["a"]
        # shared_dict["recurrent_population_params"]["b"] = model_dict["model_state_dict"]["SE_adRLIF"]["b"]
        shared_dict["recurrent_population_params"]["weights"] = [model_dict["model_state_dict"]["SE_adRLIF"]["ff_weights"], model_dict["model_state_dict"]["SE_adRLIF"]["rec_weights"]]
        
        shared_dict["decoder_population_params"]["weights"] = model_dict["model_state_dict"]["LI"]["ff_weights"]
        shared_dict["decoder_population_params"]["beta"] = model_dict["model_state_dict"]["LI"]["beta"]
        
        return shared_dict

    # --------------------------
    # Results saving
    # --------------------------
    def save_results(self, model, results, name="test_results", spikes_rasters=None):
        """Save training/testing results, optionally with spike rasters separately."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # save main results (losses, output, target) compressed
            main_path = os.path.join(self.models_dir, f"{model.name}_{name}.npz")
            results_np = {
                k: v.detach().cpu().numpy() if hasattr(v, 'detach') else np.array(v)
                for k, v in results.items()
            }
            np.savez_compressed(main_path, **results_np)
            self.results_path = main_path
            self.logger.info(f"Results saved to {main_path}")
            print(f"Time {timestamp}: Results saved to {main_path}")

            # save rasters separately if provided
            if spikes_rasters is not None:
                raster_path = os.path.join(self.models_dir, f"{model.name}_{name}_rasters.npz")
                raster_array = np.array(spikes_rasters, dtype=bool)
                np.savez_compressed(raster_path, spikes_rasters=raster_array)
                self.rasters_path = raster_path
                self.logger.info(f"Rasters saved to {raster_path}")
                print(f"Time {timestamp}: Rasters saved to {raster_path}")

            return main_path

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}", exc_info=True)
            raise 

    # --------------------------
    # Figure saving
    # --------------------------
    def save_figures(self, model, name):
        """Save plots and figures from results."""
        try:
            figures_directory = os.path.join(self.figures_dir, model.name, name)
            os.makedirs(figures_directory, exist_ok=True)

            # load results — support both old .pt and new .npz format
            if self.results_path.endswith(".npz"):
                f = np.load(self.results_path, allow_pickle=True)
                results = {k: f[k].tolist() for k in f.files}
            else:
                results = torch.load(self.results_path, weights_only=False)

            plot(results["losses"], results["output"], results["target"], model.layer2, model.layer3, figures_directory=figures_directory, save=True)
            self.logger.info(f"Figures saved to {figures_directory}")
        except Exception as e:
            self.logger.error(f"Failed to save figures: {str(e)}", exc_info=True)
            raise
    
    # --------------------------
    # Logging utilities
    # --------------------------
    def log_info(self, message):
        """Log an info-level message."""
        self.logger.info(message)
    
    def log_debug(self, message):
        """Log a debug-level message."""
        self.logger.debug(message)
    
    def log_warning(self, message):
        """Log a warning-level message."""
        self.logger.warning(message)
    
    def log_error(self, message, exc_info=False):
        """Log an error-level message."""
        self.logger.error(message, exc_info=exc_info)
        


