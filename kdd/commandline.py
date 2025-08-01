import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--resplit', help = 'whether to merge the train and test sets and resplit randomly, retaining the attack type distribution.', default = 'False')
parser.add_argument('--qthreshold', help = 'the quantile threshold above which the reconstruction loss is deemed anomalous', default = '0.99')
parser.add_argument('--log', help = 'logging level', default = 'INFO')

args = parser.parse_args()


#resplit
if not args.resplit in ['True', 'False']:
    raise ValueError('\'resplit\' must be \'True\' or \'False\'.')
exec('resplit = {}'.format(args.resplit))    # The 'exec' works properly only in the global scope.

#q_threshold
exec('q_threshold = {}'.format(args.qthreshold))

#logging_level
logging_level = args.log
