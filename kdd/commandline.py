import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--resplit', help = 'whether to merge the train and test sets and resplit randomly, retaining the attack type distribution.', default = 'False')
parser.add_argument('--qthreshold', help = 'the quantile threshold above which the reconstruction loss is deemed as anomalous', default = '0.99')
parser.add_argument('--log', help = 'logging level', default = 'INFO')

args = parser.parse_args()

if args.resplit == 'True':
    resplit = True
elif args.resplit == 'False':
    resplit = False
q_threshold = float(args.qthreshold)
logging_level = args.log
