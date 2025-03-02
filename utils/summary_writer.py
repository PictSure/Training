import re
import os
from datetime import datetime
from collections import defaultdict
import time
import json
import csv
import torch
from pprint import pprint

class SummaryWriter:
    def __init__(self, directory="./runs/", runname=None, runfolder=None):
        """
        Initialize the SummaryWriter with metrics to track.
        
        :param metrics: Names of metrics to track as strings.
        """
        self.pardir = directory
        if runfolder:
            self.runname = runfolder
        else:
            self.runname = runname + "_" + datetime.now().strftime(
                '%Y%m%d_%H%M%S') if runname else datetime.now().strftime('%Y%m%d_%H%M%S')
        self.rundir = os.path.join(
            directory, self.runname)
        self.params = dict()
        self.batch_csv_path = os.path.join(
            self.rundir, "batch_metrics.csv"
        )
        self.epoch_csv_path = os.path.join(
            self.rundir, "epoch_metrics.csv"
        )
        # Internal caches to hold metrics until flush
        # list of (phase, batch_idx, metric_name, metric_value)
        self.batch_cache = []
        # list of (phase, epoch_idx, metric_name, metric_value)
        self.epoch_cache = []

        self.make_run_dir()
    
    def make_run_dir(self):
        if not os.path.exists(self.rundir):
            os.makedirs(self.rundir, exist_ok=True)
        print(f"Run directory: {self.rundir}")

    
    def log_hyperparameters(self, params):
        self.params = params
        self._save_params()
    
    def _save_params(self):
        with open(os.path.join(self.rundir, "hyperparameters.json"), "w") as f:
            json.dump(self.params, f, indent=4)

    def log_batch_metrics(self, phase: str, batch_idx: int, metrics_dict: dict):
        """
        Log batch-level metrics in memory (and store them in an internal cache) for later disk write.
        Args:
            phase: 'train', 'val', 'test', etc.
            batch_idx: index of the current batch
            metrics_dict: dictionary of metric_name -> value
        """
        for k, v in metrics_dict.items():
            self.batch_cache.append((phase, batch_idx, k, v))

    def log_epoch_metrics(self, phase: str, epoch_idx: int, metrics_dict: dict):
        """
        Log epoch-level metrics in memory (and store them in an internal cache) for later disk write.
        Args:
            phase: 'train', 'val', 'test', etc.
            epoch_idx: index of the current epoch
            metrics_dict: dictionary of metric_name -> value
        """
        for k, v in metrics_dict.items():
            self.epoch_cache.append((phase, epoch_idx, k, v))

    def save_figure(self, figure, filename):
        figure.savefig(os.path.join(self.rundir, filename))
    
    def save_model(self, model: torch.nn.Module, epoch_idx: int = None, filename: str = None):
        """
        Save a PyTorch model checkpoint to disk.
        Args:
            model: PyTorch model
            epoch_idx: Current epoch index
            filename: Custom filename (otherwise uses "model_epoch_{epoch_idx}.pth")
        """
        if filename is None:
            filename = f"model_epoch_{epoch_idx}.pth"
        save_path = os.path.join(self.rundir, filename)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def save_checkpoint(self, checkpoint: dict, filename: str = "checkpoint.pt"):
        save_path = os.path.join(self.rundir, filename)
        torch.save(checkpoint, save_path)
    
    def flush(self):
        is_new_file = not os.path.exists(self.batch_csv_path)
        with open(self.batch_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(
                    ["phase", "batch_idx", "metric_name", "metric_value"])

            while self.batch_cache:
                phase, batch_idx, metric_name, metric_value = self.batch_cache.pop(
                    0)
                writer.writerow(
                    [phase, batch_idx, metric_name, metric_value])
        is_new_file = not os.path.exists(self.epoch_csv_path)
        with open(self.epoch_csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(
                    ["phase", "epoch_idx", "metric_name", "metric_value"])

            while self.epoch_cache:
                phase, epoch_idx, metric_name, metric_value = self.epoch_cache.pop(
                    0)
                writer.writerow(
                    [phase, epoch_idx, metric_name, metric_value])

    def close(self):
        self.flush()
        print("SummaryWriter closed.")
        

def find_latest_run_directory(base_output_dir, runname):
    """
    Finds the latest run directory matching the pattern '[runname]_[YYYYMMDD_HHMMSS]'.

    Args:
        base_output_dir (str): The parent directory where runs are stored.
        runname (str): The base name of the run (e.g., experiment name).

    Returns:
        str: The path to the latest run directory or None if no match is found.
    """
    # Regex pattern for directories in the format 'runname_YYYYMMDD_HHMMSS'
    pattern = re.compile(rf"^{re.escape(runname)}_\d{{8}}_\d{{6}}$")

    matching_dirs = [
        d for d in os.listdir(base_output_dir)
        if pattern.match(d) and os.path.isdir(os.path.join(base_output_dir, d))
    ]

    if not matching_dirs:
        return None  # No previous runs found

    # Sort by date-time extracted from the directory names
    matching_dirs.sort(
        key=lambda x: datetime.strptime(x[len(runname) + 1:], "%Y%m%d_%H%M%S"),
        reverse=True
    )

    return matching_dirs[0]


class Timer:
    def __init__(self):
        """Initialize the timer."""
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer and calculate elapsed time."""
        if self.start_time is None:
            raise RuntimeError("Timer not started.")
        self.end_time = time.time()
        return self.elapsed_time()
    
    def now(self):
        if self.start_time is None:
            raise RuntimeError("Timer not started.")
        return time.time() - self.start_time

    def elapsed_time(self):
        """Return the elapsed time."""
        if self.start_time is None or self.end_time is None:
            raise RuntimeError("Timer has not been run.")
        return self.end_time - self.start_time
    
    def reset(self):
        self.start_time = None
        self.end_time = None
            