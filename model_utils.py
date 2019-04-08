import torchvision.transforms as transforms
import torchvision.datasets
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

import os
import matplotlib.pyplot as plt

def load_data(breeds, train_transforms, val_transforms, batch_size):
    train_path = os.path.join('data/masked_images/data/train/', breeds)
    val_path = os.path.join('data/masked_images/data/val', breeds)
    train_data = torchvision.datasets.ImageFolder(train_path,
                    transform=train_transforms)
    val_data = torchvision.datasets.ImageFolder(val_path,
                    transform=val_transforms)

    img, label = train_data[300]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    classes = train_data.classes
    return train_loader, val_loader, classes



def load_split_train_val(datadir, train_transforms, val_transforms, normalize, valid_size = .25):

    train_data = torchvision.datasets.ImageFolder(datadir,
                    transform=train_transforms)
    val_data = torchvision.datasets.ImageFolder(datadir,
                    transform=val_transforms)
    num_train = len(train_data) # train + val
    indices = list(range(num_train))
    classes = train_data.classes
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    trainloader = torch.utils.data.DataLoader(train_data,sampler=train_sampler, batch_size=1)
    valloader = torch.utils.data.DataLoader(val_data,sampler=val_sampler, batch_size=1)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    return trainloader, valloader, classes, labels


def train_model(model, criterion, optimizer, scheduler, trainloader, valloader, num_epochs=5):
    dataset_sizes = {
        "train": len(trainloader),
        "val": len(valloader)
    }
    best_model_wts = model.state_dict()
    best_acc = 0.0

    use_gpu = torch.cuda.is_available()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
                dataloader = trainloader
            else:
                model.train(False)
                dataloader = valloader
            running_loss = 0.0
            running_corrects = 0.0

            for data in dataloader:
                inputs, labels = data

                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print('-' * 20)
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

def print_test_acc(model, test_transforms, normalize, test_data_dir):
    
    test_data = torchvision.datasets.ImageFolder(test_data_dir, transform=test_transforms)
    classes = test_data.classes
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1)
    total_images = 0
    total_correct = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    use_gpu = torch.cuda.is_available()

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if use_gpu:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            label = labels[0]
            class_correct[label] += c.item()
            class_total[label] += 1
            total_images += 1
            total_correct += c.item()
    print('Overall accuracy for all breeds %2d%% ' % (
            100 * total_correct / total_images))
    print()
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
