from pytorch_lightning.callbacks import Callback

class TrainingResolutionAnnealing(Callback):
    def __init__(self, resolution_update_epochs, resolution_list):
        self.update_epochs = resolution_update_epochs
        self.resolution_list = resolution_list

    def on_epoch_start(self, trainer, pl_module):
        if pl_module.current_epoch in self.update_epochs:
            resolution_index = self.update_epochs.index(
                    pl_module.current_epoch)+1
            resolution = self.resolution_list[resolution_index]
            if pl_module.training_resolution != resolution:
                print(f"Updating resolution from {pl_module.training_resolution} to {resolution}")
                pl_module.discriminator.increase_resolution_()
                pl_module.training_resolution = resolution
