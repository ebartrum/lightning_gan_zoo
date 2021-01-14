from pytorch_lightning import loggers

class CustomTensorBoardLogger(loggers.TensorBoardLogger):
    def log_metrics(self, metrics=None, step=None):
        metrics.pop('epoch', None)
        super().log_metrics(metrics, step)
