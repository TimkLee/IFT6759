from torchvision.models import resnet18
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.utils.data import DataLoader, TensorDataset
import wandb

class ModelClass(nn.Module):


    def __init__(self, device, num_classes=10, optimizer = "adam", lr=0.003, weight_decay = 0.01, momentum = 0.9,criterion = "CrossEntropyLoss"):
        """
        Reference : https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/
        """
        super(ModelClass,self).__init__()
        
        self.num_classes = num_classes
        self.device = device

        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(ResidualBlock, out_channels=16, blocks = 3)
        self.layer2 = self.make_layer(ResidualBlock, out_channels=32, blocks = 3, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, out_channels=64, blocks = 3, stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, self.num_classes)        
        
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=lr,momentum = momentum, weight_decay=weight_decay)
        else:
            print("Optimizer not recognized")

        if criterion == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()    
        else:
            print("Loss criterion not recognized")
            
    
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input : x of shape (batch size, channel , height, width)
        output : y_pred, output after forward pass into resnet18 block. With shape 
        """

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def train_sup_up(self, data, target):
        """
        TRAIN LOOP FOR SUPERVISED/SEMI-SUPERVISED LEARNING
        Train the model with the given batched data sample and targets.
        """
        
        #self.train()
        self.optimizer.zero_grad()
        y_pred = self.forward(data)
        loss = self.criterion(y_pred,target)
        loss.backward()
        self.optimizer.step()
        #acc = (torch.argmax(y_pred, dim=1) == torch.argmax(target, dim=1)).float().sum()/target.shape[0]      
        #print(loss.item())
        return loss.item()

    
    def train_shot(self, epoch, dataloader, optimizer, criterion):
        """
        TRAIN LOOP FOR FEWSHOT/ZEROSHOT LEARNING
        Train the model with the given criterion, optimizer, dataloader for given epochs. Results propogated to WANDB for visualization
        """
        #TODO
        pass
  
    @torch.no_grad()
    def test(self, dataloader):
        """
        Calculate the Loss and Accuracy of the model on given dataloader
        """
        self.eval()
        loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss(reduction="sum") #To sum up Batch loss 
        for data, targets in dataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.forward(data)
            loss += criterion(outputs, targets).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum()

        loss /= len(dataloader.dataset)
        accuracy = 100. * correct / len(dataloader.dataset)

        return loss, accuracy

    @torch.no_grad()
    def evaluation(self, test_loader, project, entity, name):
        """
        Evaluate the model for various evaluation metrics. Results propogated to WANDB for visualization 
        """
        with wandb.init(project=project, entity=entity, job_type="report", name=name) as run:
            #Class Names
            classes = tuple(test_loader.dataset.classes)

            #Test Loss and Accuracy #Eval 1
            test_loss, test_accuracy = self.test(test_loader)
            run.summary.update({"test/loss": test_loss, "test/accuracy": test_accuracy})
            wandb.log({"test/loss": test_loss, "test/accuracy": test_accuracy})
            
            #Per Class Accuracy #Eval 2
            correct_pred, total_pred = self.per_class_accuracy(test_loader) 
            columns = ["Configs"]
            accuracies = [name] #Replace it with name of run which would be equal to Config1 and so on.

            for classname, correct_count in correct_pred.items():
                accuracy = 100. * correct_count / total_pred[classname]
                columns.append(classname)
                accuracies.append(accuracy)
                #print(f'Accuracy for class: {classname} is {accuracy:.1f} %')
            #print(columns, accuracies)
            tbl = wandb.Table(columns=columns)
            tbl.add_data(*accuracies)
            wandb.log({"Per class Accuracy": tbl})

            #Extraction of Dataset from Dataloader and convertion into TensorDataset for Eval3 and Eval 4
            testset = self.tensor_dataset(test_loader)

            #K-hardest examples #Eval 3
            highest_losses, hardest_examples, true_labels, predictions = self.get_hardest_k_examples(testset)
            #print("eval:", predictions)
            wandb.log({"high-loss-examples":
                    [wandb.Image(hard_example, caption= "Pred: " + str(classes[int(pred)]) + ", Label: " +  str(classes[int(label)]))
                        for hard_example, pred, label in zip(hardest_examples, predictions, true_labels)]})
            
            #Confusion Matrix #Eval 4
            labels, predictions = self.confusion_matrix(testset)
            # print("conf:", labels)
            # print("conf:", predictions)
            wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None, preds=predictions, y_true=labels, class_names=classes)})

    @torch.no_grad()
    def per_class_accuracy(self, dataloader):
        """
        Calculate Per Class Accuracy of the model on given dataloader
        """
        classes = tuple(dataloader.dataset.classes)
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        self.eval()
        for data, targets in dataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.forward(data)
            _, predictions = torch.max(outputs, dim=1)
            #Collect Correct Predictions for each class
            for target, prediction in zip(targets, predictions):
                if target == prediction:
                    correct_pred[classes[target]] += 1
                total_pred[classes[target]] += 1

        return correct_pred, total_pred

    def tensor_dataset(self, dataloader):
        """
        Utility function which converts given DataLoader's Dataset into TensorDataset
        """
        x, y = torch.tensor(dataloader.dataset.data, dtype=torch.float32), torch.tensor(dataloader.dataset.targets)
        x = x.swapaxes(2, 3)
        x = x.swapaxes(1, 2)
        return TensorDataset(x, y)

    @torch.no_grad()
    def get_hardest_k_examples(self, dataset, k=32):
        """
        Finds the K-Hardest Examples fromt the given TensorDataset
        NB: Please pass the dataloader into the tensor_dataset function to get the appropriate TensorDataset
        """
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        losses = None
        predictions = None

        self.eval()
        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.forward(data)
            print('Outputs:', outputs)
            loss = self.criterion(outputs, targets)
            pred = outputs.argmax(dim=1, keepdim=True)
            print('Pred:', pred)
            print('Label', targets)
            if losses is None:
                losses = loss.view((1, 1))
                predictions = pred
            else:
                losses = torch.cat((losses, loss.view((1, 1))), dim=0)
                predictions = torch.cat((predictions, pred), dim=0)

        argsort_loss = torch.argsort(losses, dim=0)

        highest_k_losses = losses[argsort_loss[-k:]]
        hardest_k_examples = dataset[argsort_loss[-k:]][0]
        true_labels = dataset[argsort_loss[-k:]][1]
        #print("fn:", predictions)
        predictions = predictions[argsort_loss[-k:]]

        return highest_k_losses, hardest_k_examples, true_labels, predictions

    @torch.no_grad()
    def confusion_matrix(self, dataset):
        """
        Returns the Class Predictions, True Predictions and Classnames for given TensorDataset
        NB: Please pass the dataloader into the tensor_dataset function to get the appropriate TensorDataset
        """
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        predictions = []
        labels = []

        self.eval()
        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.forward(data)
            pred = outputs.argmax(dim=1)

            labels.append(targets.cpu().item())
            predictions.append(pred.cpu().item())

        return labels, predictions
    
    def update_lr(self, lr):    
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
