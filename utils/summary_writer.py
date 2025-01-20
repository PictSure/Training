import os
from datetime import datetime
from collections import defaultdict
import time
import json
import torch

class SummaryWriter:
    def __init__(self, directory="./runs/", metrics=[]):
        """
        Initialize the SummaryWriter with metrics to track.
        
        :param metrics: Names of metrics to track as strings.
        """
        self.metrics = set(metrics)
        self.pardir = directory
        self.rundir = os.path.join(
            directory, datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.params = dict()
        self.make_run_dir()
        self.reset()
    
    def make_run_dir(self):
        if not os.path.exists(self.rundir):
            os.makedirs(self.rundir)

    def reset(self):
        """Reset all tracked metrics."""
        self.epoch_metrics = {metric: [] for metric in self.metrics}
        self.batch_metrics = defaultdict(list)
    
    def log_hyperparameters(self, params):
        self.params = params

    def log_batch_metric(self, metric_name, value):
        """Log a batch-level metric."""
        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' is not being tracked.")
        self.batch_metrics[metric_name].append(value)

    def log_epoch_metric(self, metric_name, value):
        """Log an epoch-level metric."""
        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' is not being tracked.")
        self.epoch_metrics[metric_name].append(value)

    def get_batch_average(self, metric_name):
        """Get the average of a batch-level metric over all logged batches in the current epoch."""
        values = self.batch_metrics[metric_name]
        return sum(values) / len(values) if values else 0

    def get_epoch_average(self, metric_name, epoch=None):
        """
        Get the value or average of an epoch-level metric.
        
        :param epoch: The specific epoch to retrieve. If None, retrieves the last logged epoch's value.
        """
        if epoch is None:
            return self.epoch_metrics[metric_name][-1] if self.epoch_metrics[metric_name] else 0
        return self.epoch_metrics[metric_name][epoch]

    def print_epoch_summary(self, epoch):
        """Print a summary of all tracked metrics for the given epoch."""
        print(f"Epoch {epoch + 1} Summary:")
        for metric in self.metrics:
            avg_value = self.get_batch_average(metric)
            print(f" - {metric}: {avg_value:.4f}")
        # Reset batch-level metrics after logging
        self.batch_metrics = defaultdict(list)

    def save_to_file(self):
        """Save the tracked metrics to a JSON file."""
        with open(os.path.join(self.rundir, "metrics.json"), 'w') as f:
            json.dump({
                'epoch_metrics': self.epoch_metrics,
                # Convert defaultdict to regular dict for serialization
                'batch_metrics': dict(self.batch_metrics)
            }, f, indent=4)
        with open(os.path.join(self.rundir, "hyperparameters.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)

    def save_figure(self, figure, filename):
        figure.savefig(os.path.join(self.rundir, filename))
    
    def save_model(self, model):
        torch.save(model.state_dict(), os.path.join(self.rundir, "model.pth"))
        

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

# Example usage
# if __name__ == "__main__":
#     tracker = SummaryWriter(['loss', 'accuracy'])
    
#     for epoch in range(10):
#         timer = Timer()
#         timer.start()
        
#         tracker.reset()  # Reset batch metrics at the start of each epoch
#         for batch in range(20):  # Assume 20 batches per epoch
#             loss = 0.5 - 0.02 * batch  # Simulated loss decreasing with batches
#             accuracy = 0.1 + 0.04 * batch  # Simulated accuracy increasing with batches
#             tracker.log_batch_metric('loss', loss)
#             tracker.log_batch_metric('accuracy', accuracy)

#         epoch_loss = tracker.get_batch_average('loss')
#         epoch_accuracy = tracker.get_batch_average('accuracy')
#         tracker.log_epoch_metric('loss', epoch_loss)
#         tracker.log_epoch_metric('accuracy', epoch_accuracy)

#         timer.stop()
#         print(f"Epoch {epoch + 1} completed in {timer.elapsed_time():.2f}s.")
#         tracker.print_epoch_summary(epoch)

#     # Save metrics to a file
#     tracker.save_to_file()
            