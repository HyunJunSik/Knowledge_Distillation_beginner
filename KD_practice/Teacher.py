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
def load_dataset():
    mean = [x/255 for x in [129.3, 124.1, 112.4]]
    std = [x/255 for x in [68.2, 65.4 ,70.4]]

    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = torchvision.datasets.CIFAR100(root='../../data', train=True, download=True, transform=transformation)
    testset = torchvision.datasets.CIFAR100(root='../../data', train=False, download=True, transform=transformation)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=4)

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
        print("Train | Loss:%.4f Acc: %.2f%% (%s/%s)"
            %(epoch_loss, epoch_acc, correct, total))
    return epoch_loss, epoch_acc


def main():
    print(f"device : {device}")
    # Load Dataset
    train_loader, test_loader = load_dataset()
    num_classes = 100
    criterion = nn.CrossEntropyLoss()
    # Pre-Trained ResNet load
    resnet_34 = models.resnet34(pretrained=True)
    optimizer = optim.SGD(resnet_34.parameters(), lr=0.001, momentum=0.9)

    '''
    Fine-Tuning을 위한 2가지 작업
    1. 기존 layer의 weight 고정(freeze)
    2. 마지막 Fully-Connected Layer를 task에 맞도록 변경(fine-tuning)

    이렇게 하면 기존의 weight들은 그대로 사용하고, 마지막 fc layer만 튜닝해서 최소한의 학습을 통해 모델링 가능
    pretrained=True로 하면 기본적으로 ImageNet에 학습된 Weight를 불러와주나, 최근에는 ImageNet 버전에 따라 weight를 불러와서 입력해주는 걸 권장
    '''
    num_ftrs = resnet_34.fc.in_features
    resnet_34.fc = nn.Linear(num_ftrs, num_classes)
    resnet_34 = resnet_34.to(device)

    
    start_time = time.time()
    logging.basicConfig(filename=f"./log/experiment_{start_time}.log", level=logging.INFO, format='%(asctime)s - %(message)s')
    best_acc = 0
    epoch_length = 150
    save_loss = {"train" : [], "test" : []}
    save_acc = {"train" : [], "test" : []}

    for epoch in range(epoch_length):
        logging.info(f"Epoch: {epoch + 1}/{epoch_length}")
        print(f"epochs : {epoch + 1} / {epoch_length}")
        train_loss, train_acc = train(resnet_34, criterion, train_loader, optimizer)
        save_loss['train'].append(train_loss)
        save_acc['train'].append(train_acc)

        test_loss, test_acc = test(resnet_34, criterion, test_loader)
        save_loss['test'].append(test_loss)
        save_acc['test'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(resnet_34.state_dict())
        resnet_34.load_state_dict(best_model_wts)
        logging.info(f"Train Loss: {train_loss}, Train Acc: {train_acc}, Test Loss: {test_loss}, Test Acc: {test_acc}")

    learning_time = time.time() - start_time
    logging.info(f"Learning Time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s")
    
    torch.save(best_model_wts, 'best_model_weights.pth')
    print(f"Learning Time : {learning_time // 60:.0f}m {learning_time % 60:.0f}s")
    

if __name__ == "__main__":
    main()
    