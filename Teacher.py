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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# We Should Fine-Tuning Model for Training Cifar100
def load_dataset():
    mean = [x/255 for x in [129.3, 124.1, 112.4]]
    std = [x/255 for x in [68.2, 65.4 ,70.4]]

    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transformation)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transformation)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=4)

    return train_loader, test_loader
 
def train(model, criterion, train_loader, optimizer):

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        print(f"batch index : {batch_idx}")
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = train_loss / total
    epoch_acc = correct / total * 100
    print(f"Train | Loss : {epoch_loss} Acc : {epoch_acc}, {correct} / {total}")

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
        print(f"Test | Loss : {epoch_loss} Acc : {epoch_acc}, {correct} / {total}")
    return epoch_loss, epoch_acc


def main():
    
    # Load Dataset
    train_loader, test_loader = load_dataset()
    num_classes = 100
    criterion = nn.CrossEntropyLoss()

    # Pre-Trained ResNet load
    resnet_34 = models.resnet34(pretrained=True,)
    optimizer = optim.SGD(resnet_34.parameters(), lr=0.001, momentum=0.9)

    # scheduler : CosineAnnealingLR 방법으로 lr을 자동으로 줄여나가며 T_max까지 최저점으로 줄여나가다가 다시 최대로 올림
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    '''
    Fine-Tuning을 위한 2가지 작업
    1. 기존 layer의 weight 고정(freeze)
    2. 마지막 Fully-Connected Layer를 task에 맞도록 변경(fine-tuning)

    이렇게 하면 기존의 weight들은 그대로 사용하고, 마지막 fc layer만 튜닝해서 최소한의 학습을 통해 모델링 가능
    pretrained=True로 하면 기본적으로 ImageNet에 학습된 Weight를 불러와주나, 최근에는 ImageNet 버전에 따라 weight를 불러와서 입력해주는 걸 권장
    '''
    
    # Freezing
    for param in resnet_34.parameters():
        param.requires_grad = False
    
    fc_in_features = resnet_34.fc.in_features
    resnet_34.fc = nn.Linear(fc_in_features, num_classes)
    resnet_34 = resnet_34.to(device)
    
    start_time = time.time()
    best_acc = 0
    epoch_length = 150
    save_loss = {"train" : [], "test" : []}
    save_acc = {"train" : [], "test" : []}

    for epoch in range(epoch_length):
        print(f"epochs : {epoch + 1} / {epoch_length}")
        train_loss, train_acc = train(resnet_34, criterion, train_loader, optimizer)
        save_loss['train'].append(train_loss)
        save_acc['train'].append(train_acc)

        test_loss, test_acc = test(resnet_34, criterion, test_loader)
        save_loss['test'].append(test_loss)
        save_acc['test'].append(test_acc)

        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(resnet_34.state_dict())
        resnet_34.load_state_dict(best_model_wts)
    learning_time = time.time() - start_time
    print(f"Learning Time : {learning_time // 60:.0f}m {learning_time % 60:.0f}s")

if __name__ == "__main__":
    main()