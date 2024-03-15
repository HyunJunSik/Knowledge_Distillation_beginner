import os
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def accuracy(output, target, topk=(1,)):
    '''
    topk = 1이라면 가장 높은 예측 확률을 가진 레이블과 실제 레이블이 동일한지 계산 
    topk = (1, 5)라면, 가장 높은 예측 확률을 가진 레이블과 실제 레이블이 동일한 경우를 계산하여
    top1 정확도 구하고, 그 다음으로 높은 5개의 예측 확률을 가진 레이블 중 실제 레이블이 포함되는지 확인하여 top5 정확도 구함
    
    더욱 모델의 성능을 상세하게 평가하기 위한 방법으로, 모델의 성능을 다각도로 이해하고 평가하는 데 도움됨
    '''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def validate(val_loader, distiller):
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    
    distiller.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            output = distiller(inputs)
            loss = criterion(output, labels)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            batch_size = inputs.size(0)
            print(f"Loss: {loss.item()}, Top-1 Accuracy: {acc1.item()}, Top-5 Accuracy: {acc5.item()}")
    
    
class BaseTrainer(object):
    def __init__(self, distiller, train_loader, val_loader):
        self.train_loader = train_loader
        self.distiller = distiller
        self.val_loader = val_loader
        self.best_acc = -1
        self.optimizer = self.init_optimizer()
        
    def init_optimizer(self):
        optimizer = optim.SGD(
            # get_learnable_parameter가 distiller에 잘 되어있는지 체크
            self.distiller.module.get_learnable_parameters(),
            lr=0.05,
            momentum=0.9,
            weight_decay=0.0001,
        )
        return optimizer
    
    def train(self):
        epoch_length = 240
        for epoch in range(epoch_length):
            self.train_epoch(epoch)
    
    def train_epoch(self, epoch):
        
        if epoch in [150, 180, 210]:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1
                
        num_iter = len(self.train_loader)
        self.distiller.train()
        for idx, (inputs, labels) in enumerate(self.train_loader):
            self.train_iter(inputs, labels, epoch)
        
        # validate
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller)

    def train_iter(self, inputs, labels, epoch):
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)
        self.optimizer.zero_grad()
        
        preds, losses_dict = self.distiller(inputs, labels, epoch)
        
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        acc1, acc5 = preds.accuracy(preds, labels, topk=(1, 5))
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Top-1 Accuracy: {acc1.item()}, Top-5 Accuracy: {acc5.item()}")
        


if __package__ is None:
    import sys
    from os import path
    print(path.dirname(path.dirname(path.abspath(__file__))))
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from Distiller import distiller, KD
    from models import resnet, vgg, wrn, shufflenet_v1
    from train import load_dataset
else:
    from ..Distiller import distiller, KD
    from ..models import resnet, vgg, wrn, shufflenet_v1
    from .train import load_dataset
    
def main(selected_teacher, selected_distiller):
    train_loader, val_loader =  load_dataset()
    model_student, student_name = resnet.resnet8(num_classes=100)
    model_teacher, teacher_name = [
        resnet.resnet32x4(num_classes=100), 
        resnet.resnet32(num_classes=100), 
        wrn.wrn_40_2(num_classes=100), 
        vgg.vgg13(num_classes=100),
        ][selected_teacher]
    
    model_teacher.load_state_dict(load_teacher_param(teacher_name))
        
    distiller = [
        KD(model_student, model_teacher),
        ][selected_distiller]
    distiller = torch.nn.DataParallel(distiller.cuda())
    
    trainer = BaseTrainer(distiller, train_loader, val_loader)
    trainer.train()
    '''
    trainer = trainer_dict[cfg.SLOVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)
    여기서 SOLVER.TRAINER는 "base" - BaseTrainer
    '''

def load_teacher_param(model_name):
    model_state_dict_path = f"model_pth/best_model_weights_{model_name}.pth"
    model_state_dict = torch.load(model_state_dict_path)
    return model_state_dict