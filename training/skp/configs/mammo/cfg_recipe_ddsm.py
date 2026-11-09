"""Active DDSM config; paths and resolution come from an explicit recipe."""
from mammoscreen_recipe import config_from_environment

cfg = config_from_environment(pretraining=True)
