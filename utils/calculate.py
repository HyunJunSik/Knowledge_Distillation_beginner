import sys
from os import path
from torchprofile import profile_macs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__=="__main__":
    if __package__ is None:
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from models import wrn, shufflenet_v1
        from train import load_dataset
    else:
        from ..models import wrn, shufflenet_v1
        from .train import load_dataset
    
    model, model_name = shufflenet_v1.ShuffleV1(num_classes=100)
    
    data_loader, _ = load_dataset(bz=1)
    data, _ = next(iter(data_loader))
    
    with open("./model_computation/model_computation.txt", "a") as f:
        f.write(f"{model_name}, {profile_macs(model, data)}, {count_parameters(model)}\n")
    