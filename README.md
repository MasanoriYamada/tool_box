# awesome tools for me

- dataloader(mnist, cifar10, svhn, dsprites, lpc)
- trainer
- logger: logging hyperparameter, model, sorce file
- logger_mlflow: Extend logger for mlflow and tensorboard
- docker files: pytorch and jupyter notebook
- run shooter: simple job scheduler on gpu
- jupyter config: auto save html file from .iynb

## how to use

see example.py



## logger_mlflow

Explain logger_mlflow because it has restrictions on its use.

mlflow has server mode and local mode.

- In local mode, path of log file is determined by args.dir.
- In server mode, the path of log file is determined by --default-artifact-root (args.dir is ignored). If the log server and the execution environment are different, you need to share the log file (model, etc.) of the execution environment to the -default-artifact-root of the server by yourself.