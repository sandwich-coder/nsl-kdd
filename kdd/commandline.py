import argparse
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required = True)
args = parser.parse_args()

config = yaml.load(
    open(args.config, 'r'),
    Loader = yaml.FullLoader,
    )

LATENT = config['latent']
RESPLIT = config['resplit']
Q_THRESHOLD = config['q_threshold']
LOGGING_LEVEL = config['logging_level']
