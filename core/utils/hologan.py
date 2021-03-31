from torch.optim.lr_scheduler import LambdaLR

def create_hologan_lr_scheduler(total_epochs, optimizer):
    def lr_lambda(epoch):
        if epoch <= total_epochs/2:
            return 1
        else:
            return 1-((epoch - total_epochs/2)/(total_epochs/2))
    return LambdaLR(optimizer, lr_lambda)
