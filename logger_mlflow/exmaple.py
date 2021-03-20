import mlflow
import torch
import torchvision
import argparse
from utils import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=128, metavar='N',
                    help='input batch size for training')
parser.add_argument('--dir', type=str, default='log', metavar='N',
                    help='decsription of test')
parser.add_argument('--description', type=str, default='description', metavar='N',
                    help='decsription of test')
args = parser.parse_args()

model = torchvision.models.resnet18()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
# mlflow tracking server URL
# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_log --host 0.0.0.0 --port 7000
#mlflow_tracking_server_url = 'http://localhost:7000'
#mlflow.set_tracking_uri(mlflow_tracking_server_url)
#client = mlflow.tracking.MlflowClient()
mlflow.set_experiment('ml exmaple')  

# training code
with mlflow.start_run():
    logging = Logger(args, args.description, use_tensorboard=True, use_mlflow=True)
    logging.add_scalar('train/loss', 1.0, 0)
    logging.add_scalar('train/loss', 0.01, 1)
    logging.add_scalar('train/loss', 0.001, 2)
    results = {}
    save_model = {'model1': model, 'model2': model}
    save_opt = {'model': opt}
    logging.log_res(results=results, model=save_model, opt=save_opt)


logging = Logger(args, args.description, use_tensorboard=True, use_mlflow=True)
logging.add_scalar('train/loss', 1.0, 0)
logging.add_scalar('train/loss', 0.01, 1)
logging.add_scalar('train/loss', 0.001, 2)

# save result
#results = {'test_loss.pickle': trainer.test_losses, ...}
#results=None
results = {}
save_model = {'model1': model, 'model2': model}
save_opt = {'model': opt}
logging.log_res(results=results, model=save_model, opt=save_opt)
