import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import wandb

class ModelClass(nn.Module):
	"""
	All CNN model C 
	ref : https://arxiv.org/pdf/1412.6806.pdf check table 2 (we replace max pooling from table 1 with conv layers with stride=2 to get AllCNN correctly)
	"""
	def __init__(self, device, num_classes=10,optimizer = "adam", lr=0.003, weight_decay = 0.01, criterion = "CrossEntropyLoss",momentum=0):
		super(ModelClass, self).__init__()
		
		self.num_classes = num_classes
		self.device = device

		#Convolutional Layers:
		#Block1
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), padding='same')
		self.conv2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), padding='same')
		#BLock2
		self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), padding='same')
		self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), padding='same')
		#Block3
		self.conv5 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), padding='same')
		#Block4
		self.conv6 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 1), padding='valid')
		#Block5
		self.conv7 = nn.Conv2d(in_channels=192, out_channels=self.num_classes, kernel_size=(1, 1), padding='valid')

		#MaxPool Layers
		#Block1
		self.max1 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
		#Block2
		self.max2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

		#GlobalAveragePooling
		#Useful Reference: https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/12
		self.gap = nn.AvgPool2d(kernel_size=(6, 6))

		#hyper parameters :

		if optimizer == "adam":
			self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
		elif optimizer == "sgd":
			self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay,momentum=momentum)
		else:
			print("Optimizer not recognized")

		if criterion == "CrossEntropyLoss":
			self.criterion = nn.CrossEntropyLoss()    
		else:
			print("Loss criterion not recognized")

	def forward(self, x):
		"""
		input : x of shape (batch size, channel , height, width)
		output after forward of dimension (batch_size , num_class)
		"""
		#Block1
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.max1(x)
		#Block2
		x = F.relu(self.conv3(x))
		x = F.relu(self.conv4(x))
		x = self.max2(x)
		#Block3
		x = F.relu(self.conv5(x))
		#Block4
		x = F.relu(self.conv6(x))
		#Block5
		x = F.relu(self.conv7(x))
		#GlobalAveragePool
		x = self.gap(x)
		x = x.flatten(start_dim=1)
		return x

	def train_sup_up(self, data, target):
	  """
	  TRAIN LOOP FOR SUPERVISED/SEMI-SUPERVISED LEARNING
	  Train the model with the given batched data sample and targets.
	  """
	  self.train()
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

	        #K-hardest examples #Eval 3
	        highest_losses, hardest_examples, true_labels, predictions = self.get_hardest_k_examples(test_loader)
	        #print("eval:", predictions)
	        wandb.log({"high-loss-examples":
	                [wandb.Image(hard_example, caption= "Pred: " + str(classes[int(pred)]) + ", Label: " +  str(classes[int(label)]))
	                    for hard_example, pred, label in zip(hardest_examples, predictions, true_labels)]})
	        
	        #Confusion Matrix #Eval 4
	        labels, predictions = self.confusion_matrix(test_loader)
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


	@torch.no_grad()
	def get_hardest_k_examples(self, dataloader, k=32):
	    """
	    Finds the K-Hardest Examples fromt the given DataLoader
	    """
	   
	    batch_size= dataloader.batch_size #useless now

	    losses = torch.Tensor([])
	    predictions = torch.Tensor([])

	    self.eval()
	    for data, targets in dataloader:
	        data, targets = data.to(self.device), targets.to(self.device)
	        outputs = self.forward(data)
	        loss = F.cross_entropy(outputs, targets,reduction='none')
	        pred = outputs.argmax(dim=1, keepdim=True)
	        if losses is None:
	            losses = loss.view((loss.shape[0], 1)).cpu()
	            predictions = pred.cpu()
	        else:
	            losses = torch.cat((losses, loss.view((loss.shape[0], 1)).cpu()), dim=0)
	            predictions = torch.cat((predictions, pred.cpu()), dim=0)

	    argsort_loss = torch.argsort(losses, dim=0)

	    highest_k_losses = losses[argsort_loss[-k:]]

	    #converting to printable image output
	    d = torch.Tensor(dataloader.dataset.data).to(self.device)
	    d = d.swapaxes(2, 3)
	    d = d.swapaxes(1, 2)


	    hardest_k_examples = d[argsort_loss[-k:]] 
	    l = torch.Tensor(dataloader.dataset.targets).to(self.device)
	    true_labels = l[argsort_loss[-k:]]
	    #print("fn:", predictions)
	    predictions = predictions[argsort_loss[-k:]]

	    return highest_k_losses, hardest_k_examples, true_labels, predictions

	@torch.no_grad()
	def confusion_matrix(self, dataloader):
	    """
	    Returns the Class Predictions, True Predictions and Classnames for given DataLoader
	    """

	    predictions = np.array([])
	    labels = np.array([])

	    self.eval()
	    for data, targets in dataloader:
	        data, targets = data.to(self.device), targets.to(self.device)
	        outputs = self.forward(data)
	        #outputs = F.softmax()
	        pred = outputs.argmax(dim=1)

	        labels = np.append(labels,targets.cpu().tolist())
	        predictions = np.append(predictions,pred.cpu().tolist())

	    return labels, predictions

	def update_lr(self, lr):    
	    for param_group in self.optimizer.param_groups:
	        param_group['lr'] = lr