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

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bz, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bz, shuffle=True, num_workers=2)

    return train_loader, test_loader
 
def train(model, criterion, train_loader, optimizer):

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # loss.item()은 loss값을 스칼라로 반환
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1) # outputs.max(1)은 각 입력 샘플에 대해 가장 큰 값과 해당 인덱스 반환
        
        # predicted.eq(labels)는 예측 클래스와 실제 클래스가 동일한지에 대한 불리언 마스크 생성
        # .sum().item()은 불리언 마스크 합계 취한 뒤, 스칼라값으로 변환하여 옳게 예측된 샘플 수 얻음
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0) # labels.size(0)은 현재 배치의 레이블 수 출력  
        
    epoch_loss = train_loss/total
    epoch_acc = correct/total*100
    print("Train | Loss:%.4f Acc: %.2f%% (%s/%s)"
          %(epoch_loss, epoch_acc, correct, total))

    return epoch_loss, epoch_acc


def test(model, criterion, test_loader):
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = test_loss / total
        epoch_acc = correct / total * 100
        print("Test | Loss:%.4f Acc: %.2f%% (%s/%s)"
            %(epoch_loss, epoch_acc, correct, total))
    return epoch_loss, epoch_acc


def main(model, model_name):
    print(f"device : {device}")
    # Load Dataset
    train_loader, test_loader = load_dataset()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0001)
    model.to(device)
    
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    start_time = time.time()
    
    logging.basicConfig(filename=f"./log/experiment_cifar-100_{now}_{model_name}.log", level=logging.INFO, format='%(asctime)s - %(message)s')
    best_acc = 0
    epoch_length = 240
    save_loss = {"train" : [], "test" : []}
    save_acc = {"train" : [], "test" : []}

    for epoch in range(epoch_length):
        
        if epoch in [150, 180, 210]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        logging.info(f"Epoch: {epoch + 1}/{epoch_length}")
        print(f"epochs : {epoch + 1} / {epoch_length}")
        train_loss, train_acc = train(model, criterion, train_loader, optimizer)
        save_loss['train'].append(train_loss)
        save_acc['train'].append(train_acc)

        test_loss, test_acc = test(model, criterion, test_loader)
        save_loss['test'].append(test_loss)
        save_acc['test'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        logging.info(f"Train Loss: {train_loss}, Train Acc: {train_acc}, Test Loss: {test_loss}, Test Acc: {test_acc}")

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
        from models import wrn, shufflenet_v1, vgg
    else:
        from ..models import wrn, shufflenet_v1, vgg
    
    model, model_name = vgg.vgg13(num_classes=100)
    main(model, model_name)