import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from .distiller import Distiller

def CLKD_loss(logits_student, logits_teacher):
    '''
    instance_distill_loss, class_distill_loss, class_correlation_loss
    '''
    # L2 norm이라 p=2라고 설정, L1 norm이면 p=1
    student_norm = F.normalize(logits_student, p=2, dim=1)
    teacher_norm = F.normalize(logits_teacher, p=2, dim=1)
    instance_loss = F.mse_loss(student_norm, teacher_norm)
    
    t_stu = torch.t(logits_student)
    t_tea = torch.t(logits_teacher)
    
    t_student_norm = F.normalize(t_stu, p=2, dim=1)
    t_teacher_norm = F.normalize(t_tea, p=2, dim=1)
    class_loss = F.mse_loss(t_student_norm, t_teacher_norm)
    
    '''
    class correlation loss
    1. Class Correlation Matrix
    2. Frobenius Norm(L2 norm)
    '''
    # 1. Class Correlation Matrix
    N, C = logits_student.shape
    
    student_mean = torch.mean(logits_student, dim=0)
    teacher_mean = torch.mean(logits_teacher, dim=0)
    
    B_s, B_t = torch.zeros((N, N)).to(device), torch.zeros((N, N)).to(device)
    for j in range(C):
        student_j = logits_student[:, j]
        diff_s = student_j - student_mean[j]
        B_s += torch.outer(torch.t(diff_s), diff_s)
        
        teacher_j = logits_teacher[:, j]
        diff_t = teacher_j - teacher_mean[j]
        B_t += torch.outer(torch.t(diff_t), diff_t)
    
    B_s /= (C-1)
    B_t /= (C-1)
    
    # 2. Frobenius Norm(L2 norm)
    diff = B_s - B_t
    diff_norm = torch.norm(diff, 'fro') # Frobenius Norm
    class_corr_loss = (1 / (C**2)) * diff_norm ** 2
    
    return instance_loss, class_loss, class_corr_loss

class CLKD(Distiller):
    '''
    
    '''
    def __init__(self, student, teacher):
        super(CLKD, self).__init__(student, teacher)
        self.lamb = 0.1
        self.mu = 0.5
        self.vu = 0.4
        self.beta = 2.0
    
    def forward_train(self, image, target, epoch):
        logits_student = self.student(image)
        with torch.no_grad():
            logits_teacher = self.teacher(image)
        
        ins_loss, cla_loss, clcor_loss = CLKD_loss(logits_student, logits_teacher)
        loss_ce = self.lamb * F.cross_entropy(logits_student, target)
        loss_kd = self.mu * (ins_loss + self.beta * cla_loss) + self.vu * clcor_loss
        losses_dict = {
            "loss_ce" : loss_ce,
            "loss_kd" : loss_kd,
        }

        return logits_student, losses_dict