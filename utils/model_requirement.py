import torch
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model_ckp(model_name):
    '''
    학습된 모델 파라미터 불러오기
    '''
    model = model_name
    model_checkpoint = "~~~~~~.pth.tar 혹은 pth"
    
    # utils.load_checkpoint(model_checkpoint, model)
    # model_size = count_parameter(model)

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint