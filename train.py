from __future__ import print_function
import argparse
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torchsample as ts 
import torch.nn.functional as F 
import os 
import torch.optim as optim
from models.cat_scratch import BreedClassifier
from torch.optim import lr_scheduler

import model_utils


def evaluate(model, split, criterion, verbose=False, n_batches=None, val_loader=None, test_loader=None):
    '''
    Compute loss on val or test data.
    '''
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    if split == 'val':
        loader = val_loader
    elif split == 'test':
        loader = test_loader
    for batch_i, batch in enumerate(loader):
        data, target = batch
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss += criterion(output, target, reduction="sum").data
        # predict the argmax of the log-probabilities
        _, pred = torch.max(output.data, 1)
        
        #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        a = torch.sum(pred == target.data).to(torch.float32)
        correct += a
        #print(correct)
        #b = pred.eq(target.data.view_as(pred)).cpu().sum()
        #print(a, b)
        #print(target.data.shape, pred.shape, target.data==pred)
        n_examples += pred.size(0)
        if n_batches and (batch_i >= n_batches):
            break
    loss /= n_examples
    acc = correct / n_examples
    if verbose:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, loss, correct, n_examples, acc))
    return loss, acc


def train(model, criterion, train_loader, val_loader, optimizer, scheduler, num_epochs, log_interval=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    location = None

    use_gpu = torch.cuda.is_available()
    if use_gpu:
    	model = model.cuda()

    for epoch in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        scheduler.step()
        # training loop
        for batch_idx, batch in enumerate(train_loader):
            
            model.train(True)
            
            inputs, labels = batch
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
        
            loss.backward()
            optimizer.step()          

            if batch_idx % log_interval == 0:
                val_loss, val_acc = evaluate(model, 'val', F.cross_entropy, val_loader=val_loader)
                train_loss = loss.data
                examples_this_epoch = batch_idx * len(inputs)
                epoch_progress = 100. * batch_idx / len(train_loader)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                epoch, examples_this_epoch, len(train_loader.dataset),
                epoch_progress, train_loss, val_loss, val_acc))
                if val_acc > best_acc:
                	best_acc = val_acc
                	best_model_wts = model.state_dict()
                	location = (epoch, batch_idx)
    print('Best val Acc: {:4f} obtained at epoch: {}, batch: {}'.format(best_acc, location[0], location[1]))
    model.load_state_dict(best_model_wts)
    return model

def main(args):
    # data_dir = './data/masked_images/data/trainval'
    # data_dir = os.path.join(data_dir, 'breeds_'+ args.breed)
    # normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                               std=[0.229, 0.224, 0.225])
    # train_transforms = transforms.Compose([transforms.Resize(224),
    #                                         #transforms.CenterCrop(224),
    #                                        transforms.ToTensor(),
    #                                        normalize
    #                                        ])
    # val_transforms = transforms.Compose([transforms.Resize(224),
    #                                        transforms.ToTensor(),
    #                                        normalize
    #                                        ])
    # train_loader, val_loader, classes, labels = load_split_train_val(data_dir, train_transforms,
    #                                             val_transforms, normalize, valid_size = .25)
    # use_gpu = torch.cuda.is_available()
    # im_size = (3, 256, 256)
    # # TODO load models
    # if args.model == 'cat_scratch':
    #     #model = models.cat_scratch.CatScratch(im_size, len(classes))
    #     model = torchvision.models.resnet50(pretrained=True)
    #     num_features = model.fc.in_features
    #     model.fc = nn.Linear(num_features, len(classes))

    # #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.weight_decay)
    # criterion = F.cross_entropy
    # if use_gpu:
    #     model = model.cuda()
    # train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, args.epochs)

    # #evaluate('test', verbose=True)
    # torch.save(model, args.model + '.pt')
    data_dir = 'data/trainval/breeds_cat/'
    test_data_dir = 'data/test/breeds_cat/'

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([transforms.Resize((299, 299)),
                                           #transforms.CenterCrop(224),
                                           #transforms.RandomResizedCrop((299, 229)),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize
                                           ])
    test_transforms = transforms.Compose([transforms.Resize((299, 299)),
                                           #transforms.CenterCrop((229, 229)),
                                           transforms.ToTensor(),
                                           normalize
                                           ])
    #trainloader, valloader, classes, labels = model_utils.load_split_train_val(data_dir, train_transforms, test_transforms, normalize, .25)
    trainloader, valloader, classes, n_train, n_val, testloader, n_test = model_utils.load_data('breeds_cat', train_transforms, test_transforms, batch_size=args.batch_size)

    im_size = (3, 299, 299)
    model = BreedClassifier(im_size, len(classes))
    #model = torchvision.models.resnet50(pretrained=True)
    #model = torchvision.models.resnet18(pretrained=False)


    #num_features = model.fc.in_features
    #model.fc = nn.Linear(num_features, len(classes))
    criterion = F.cross_entropy

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.weight_decay)

    device = torch.device("cuda:5")
    model.to(device)
    model = train(model, criterion, trainloader, valloader, optimizer, scheduler, args.epochs, log_interval=10)

    save_path = './trained_model/cat_scratch.pth'
    torch.save(model, save_path)

    test_loss, acc = evaluate(model, 'test', criterion, test_loader=testloader)
    print("Test Acc: %0.4f" % acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='One Run of Experiment')
    parser.add_argument('--breed', type=str, metavar='dataset',
                        choices=['cat', 'dog'],
                        help='dataset to train, cats or dogs')
    parser.add_argument('--lr', type=float, metavar='LR',
                    help='learning rate')
    parser.add_argument('--momentum', type=float, metavar='M',
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Weight decay hyperparameter')
    parser.add_argument('--batch-size', type=int, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--model',
                        choices=['dog_scratch', 'cat_scratch', 'dog_finetune', 'cat_finetune'],
                        help='which model to train/evaluate')
    args = parser.parse_args()
    main(args)
