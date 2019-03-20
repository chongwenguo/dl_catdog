import torch 
from datagen import MyDataset
import os
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def load_split_train_test(data_dir, valid_size = .2):

    cur_dir = os.path.dirname(__file__)
    datadir = os.path.join(cur_dir, data_dir)

    normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
    train_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor(),
                                       normalize
                                       ])
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      normalize
                                      ])
    train_data = torchvision.datasets.ImageFolder(datadir,
                    transform=train_transforms)
    test_data = torchvision.datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=1)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=1)
    return trainloader, testloader



if __name__ == '__main__':
    datadir = 'data/species/'

    trainloader, testloader = load_split_train_test(datadir, .2)
    #print(trainloader.dataset.classes)


    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    model = torchvision.models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False


    model.fc = nn.Sequential(nn.Linear(2048, 512),
                             nn.ReLU(),
                             nn.Dropout(0.2),
                             nn.Linear(512, 10),
                             nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)


    epochs = 1
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss / len(trainloader))
                test_losses.append(test_loss / len(testloader))
                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")
                running_loss = 0
                model.train()
    torch.save(model, 'aerialmodel.pth')

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

    #
    # normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
    # transform=transforms.Compose([
    #     transforms.Resize((300)),#缩放图片，保持长宽比不变，最短边的长为300像素,
    #     transforms.CenterCrop(300), #从中间切出 300*300的图片
    #     transforms.ToTensor(),
    #     normalize
    # ])

    # dataset = torchvision.datasets.ImageFolder(DATA_PATH, transform=transform)


    #
    # dataloader = DataLoader(dataset,  shuffle=True, num_workers=0, drop_last=False)
    #
    # for batch_datas, batch_labels in dataloader:
    #     print(batch_datas.size(),batch_labels.size())

'''
data_transforms = {
    'train.py': transforms.Compose(
        [transforms.Resize((176,176)),
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]),
    'val': transforms.Compose(
        [transforms.Resize((128,128)),
         transforms.Grayscale(),
         transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ]),
}



train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)



data_dir = 'dataset'
image_datasets = {x: torchvision.datasets.ImageFolder(TRAIN_DATA_PATH,data_transforms[x])
                  for x in ['train.py', 'val']}


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train.py', 'val']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['train.py', 'val']}
class_names = image_datasets['train.py'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
