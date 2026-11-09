"""Local Lightning logger compatible with the preserved runner's channel API."""

import json
import os
import time
from pathlib import Path

from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.pytorch.utilities.rank_zero import rank_zero_only


def serializable(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


class Channel:
    def __init__(self, experiment, key):
        self.experiment, self.key = experiment, key

    def append(self, value, **kwargs):
        self.experiment.write('append', self.key, value, **kwargs)


class LocalExperiment:
    def __init__(self, directory):
        self.directory = Path(directory)

    def write(self, kind, key, value, **metadata):
        self.directory.mkdir(parents=True, exist_ok=True)
        event = dict(time=time.time(), kind=kind, key=key, value=value, **metadata)
        with (self.directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, default=serializable) + '\n')

    def __getitem__(self, key):
        return Channel(self, key)

    def __setitem__(self, key, value):
        self.write('assign', key, value)

    def stop(self):
        self.write('stop', '', None)


class LocalTrainingLogger(Logger):
    """Write rank-zero metrics/config to each job's logs/events.jsonl."""

    def __init__(self, project=None, **kwargs):
        super().__init__()
        self._directory = os.environ['MAMMOSCREEN_LOG_DIR']
        self._name = project or 'mammoscreen-local'
        self._experiment = None

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return 'local'

    @property
    def save_dir(self):
        return self._directory

    @property
    @rank_zero_experiment
    def experiment(self):
        if self._experiment is None:
            self._experiment = LocalExperiment(self._directory)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        self.experiment['hyperparameters'] = vars(params) if hasattr(params, '__dict__') else params

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for key, value in metrics.items():
            self.experiment[key].append(value, step=step)

    @rank_zero_only
    def finalize(self, status):
        self.experiment.write('finalize', '', status)
