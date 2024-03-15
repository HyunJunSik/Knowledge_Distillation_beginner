import torch
import torch.nn as nn
import torch.nn.functional as F

from .distiller import Distiller

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

class KD(Distiller):
    '''
    Distilling the Knowledge in a Neural Network
    KD CFG
    CFG.KD = CN()
    CFG.KD.TEMPERATURE = 4
    CFG.KD.LOSS = CN()
    CFG.KD.LOSS.CE_WEIGHT = 0.1
    CFG.KD.LOSS.KD_WEIGHT = 0.9
    '''

    def __init__(self, student, teacher):
        super(KD, self).__init__(student, teacher)
        self.temperature = 4
        self.ce_loss_weight = 0.1
        self.kd_loss_weight = 0.9
    
    def forward_train(self, image, target, **kwargs):
        logits_student = self.student(image)
        with torch.no_grad():
            logits_teacher = self.teacher(image)
        
        # loss
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.temperature)
        losses_dict = {
            "loss_ce" : loss_ce,
            "loss_kd" : loss_kd,
        }

        return logits_student, losses_dict