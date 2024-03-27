import torch 
import torch.nn as nn
import torch.nn.functional as F

from .distiller import Distiller

def GCKD_loss(logits_student, logits_teacher):
    '''
    gram matrix를 batch크기만큼의 행렬로 생성하여 
    teacher / student 간의 matrix를 줄여나가는 방향
    MSE Loss를 통해 행렬의 representation gap을 줄여나가는쪽으로
    '''
    # Normalization을 해야하나?...
    # Class Correlation쪽은 하는게 좋다고 보는데
    _, C = logits_student.size()
    t_stu = torch.t(logits_student)
    t_tea = torch.t(logits_teacher)
    
    t_student_norm = F.normalize(t_stu, p=2, dim=1)
    t_teacher_norm = F.normalize(t_tea, p=2, dim=1)

    '''
    def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv) 이거 함 넣어보자..!
    '''
    
    # Student's gram matrix
    student_gram = torch.mm(logits_student, t_student_norm)
    student_gram /= (C-1)
    
    # Teacher's gram matrix
    teacher_gram = torch.mm(logits_teacher, t_teacher_norm)
    teacher_gram /= (C-1)
    
    diff = student_gram - teacher_gram
    diff_norm = torch.norm(diff, 'fro')
    loss = (1 / (C ** 2)) * diff_norm ** 2
    
    return loss
    
class GCKD(Distiller):
    '''
    Proposed Gramian Class Knowledge Distillation
    lamb : 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    beta : 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    '''
    def __init__(self, student, teacher):
        super(GCKD, self).__init__(student, teacher)
        self.lamb = 0.8
        self.beta = 0.2
        
    def forward_train(self, image, target, epoch):
        logits_student = self.student(image)
        with torch.no_grad():
            logits_teacher = self.teacher(image)
        
        loss_kd = self.beta * GCKD_loss(logits_student, logits_teacher)
        loss_ce = self.lamb * F.cross_entropy(logits_student, target)
        
        losses_dict = {
            "loss_ce" : loss_ce,
            "loss_kd" : loss_kd,
        }
        
        return logits_student, losses_dict

    
    
    
    
    
    
    
    
    
