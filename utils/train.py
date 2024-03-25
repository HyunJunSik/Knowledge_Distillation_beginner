import os
import torch
import torch.nn as nn
import math
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import time
import copy
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

# We Should Fine-Tuning Model for Training Cifar100
def load_dataset(bz=64):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    trainset = torchvision.datasets.CIFAR100(root='./../../data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root='./../../data', train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bz, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bz, shuffle=True, num_workers=0)

    return train_loader, test_loader
 
def train(model, criterion, train_loader, optimizer):

    model.train()
    total_loss = 0
    total_acc1 = 0
    total_acc5 = 0
    dataset_len = len(train_loader.dataset)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = inputs.size(0)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()    
        # loss.item()은 loss값을 스칼라로 반환
        # _, predicted = outputs.max(1) outputs.max(1)은 각 입력 샘플에 대해 가장 큰 값과 해당 인덱스 반환
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        total_loss += loss.item() * batch_size
        total_acc1 += acc1.item() * batch_size
        total_acc5 += acc5.item() * batch_size      
          
    train_loss = total_loss / dataset_len
    train_acc1 = total_acc1 / dataset_len
    train_acc5 = total_acc5 / dataset_len
        
    return train_loss, train_acc1, train_acc5


def test(model, criterion, test_loader):
    
    model.eval()
    total_loss = 0
    total_acc1 = 0
    total_acc5 = 0
    dataset_len = len(test_loader.dataset)
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # _, predicted = outputs.max(1)
            
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_acc1 += acc1.item() * batch_size
            total_acc5 += acc5.item() * batch_size
    val_loss = total_loss / dataset_len
    val_acc1 = total_acc1 / dataset_len
    val_acc5 = total_acc5 / dataset_len
    return val_loss, val_acc1, val_acc5


def main(model, model_name):
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    print(f"device : {device}")
    # Load Dataset
    train_loader, test_loader = load_dataset()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)
    model.to(device)
    
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    start_time = time.time()
    
    logging.basicConfig(filename=f"./model_train_log/experiment_cifar-100_{now}_{model_name}.log", level=logging.INFO, format='%(asctime)s - %(message)s')
    best_acc = 0
    epoch_length = 240

    for epoch in range(epoch_length):
        
        if epoch in [150, 180, 210]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        logging.info(f"Epoch: {epoch + 1}/{epoch_length}")
        print(f"epochs : {epoch + 1} / {epoch_length}")
        train_loss, train_acc1, train_acc5 = train(model, criterion, train_loader, optimizer)

        test_loss, test_acc1, test_acc5 = test(model, criterion, test_loader)


        if test_acc1 > best_acc:
            best_acc = test_acc1
            best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        
        print(f"Train Loss: {train_loss}, Top-1 Accuracy: {train_acc1}, Top-5 Accuracy: {train_acc5}")
        logging.info(f"Train Loss: {train_loss}, Top-1 Accuracy: {train_acc1}, Top-5 Accuracy: {train_acc5}")
        
        print(f"Test Loss: {test_loss}, Top-1 Accuracy: {test_acc1}, Top-5 Accuracy: {test_acc5}")
        logging.info(f"Test Loss: {test_loss}, Top-1 Accuracy: {test_acc1}, Top-5 Accuracy: {test_acc5}")
 
    learning_time = time.time() - start_time
    logging.info(f"Learning Time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s")
    
    torch.save(best_model_wts, f"model_pth/best_model_weights_{model_name}.pth")
    print(f"Learning Time : {learning_time // 60:.0f}m {learning_time % 60:.0f}s")
    

if __name__ == "__main__":
    if __package__ is None:
        import sys
        from os import path
        print(path.dirname(path.dirname(path.abspath(__file__))))
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from models import wrn, shufflenet_v1, vgg, resnet
    else:
        from ..models import wrn, shufflenet_v1, vgg, resnet
    
    model, model_name = wrn.wrn_16_2(num_classes=100)
    main(model, model_name)