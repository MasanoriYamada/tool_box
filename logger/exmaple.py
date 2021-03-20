import argparse
from utils import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--description', type=str, default='description', metavar='N',
                    help='decsription of test')
args = parser.parse_args()


logging = Logger(args, args.description)

# training code


# save result
results = {'test_loss.pickle': trainer.test_losses, ...}
save_model = {'model': trainer.best_model}
save_opt = {'model': opt}
logging.log_res(results=results, model=save_model, opt=save_opt)
