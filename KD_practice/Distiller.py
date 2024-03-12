import torch
import torch.nn as nn
import torch.nn.functional as F

class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        # Teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self
    
    def get_learnable_parameter(self):
        return [v for k, v in self.student.named_parameters()]
    
class Vanilla(nn.Module):
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student
    
    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, {"ce" : loss}
    
