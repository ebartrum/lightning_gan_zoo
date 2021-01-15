from pytorch_lightning.callbacks import Callback
from hydra.utils import instantiate
from . import types

class FiguresCallback(Callback):
    def __init__(self, figs):
        super(FiguresCallback, self).__init__()
        self.figs = [instantiate(f) for f in figs]

    def on_validation_end(self, trainer, pl_module):
        print("Drawing & saving figures...")
        for fig in self.figs:
            print(f"Drawing & saving {fig.filename}...")
            fig.draw_and_save(pl_module)
