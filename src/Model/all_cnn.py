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
	def __init__(self, num_classes=10,optimizer = "adam", lr=0.003, weight_decay = 0.01, criterion = "CrossEntropyLoss", device):
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

	def train_sup_up(self, train_loader, valid_loader, epochs, batch_log_interval=25):
		"""
		TRAIN LOOP FOR SUPERVISED/SEMI-SUPERVISED LEARNING
		Train the model with the given device, train_loader for given epochs. Results propogated to WANDB for visualization after batch_log_intervals and end of epoch
		"""
		with wandb.init(project="Supervised Learning", entity='ift6759-aiadlp', job_type="train") as run:
			self.train()
			example_ct = 0
			for epoch in range(epochs): 
				for batch_idx, (data, targets) in enumerate(train_loader):
					data, targets = data.to(self.device), targets.to(self.device)
					self.optimizer.zero_grad()
					outputs = self.forward(data)
					loss = self.criterion(outputs, targets)
					loss.backward()
					self.optimizer.step()

					example_ct += len(data)

					if batch_idx % batch_log_interval == 0:
						#Logging into WANDB
						print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

						loss = float(loss)
						wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
						print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

				#Logging into WANDB
				train_loss, train_accuracy = self.test(train_loader)
				valid_loss, valid_accuracy = self.test(valid_loader)
				loss, accuracy = float(loss), float(accuracy)
				wandb.log({"epoch": epoch, "train/loss": train_loss, "train/accuracy": train_accuracy, "validation/loss": valid_loss, "validation/accuracy": valid_accuracy}, step=example_ct)	
				print(f"Train Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {train_loss:.3f}/{train_accuracy:.3f}")
				print(f"Validation Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {valid_loss:.3f}/{valid_accuracy:.3f}")
		      
		torch.save(self.state_dict(), "trained_model.pth")
		#return acc, loss

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
	def evaluation(self, test_loader):
		"""
		Evaluate the model for various evaluation metrics. Results propogated to WANDB for visualization 
		"""
		with wandb.init(project="Supervised Learning", entity='ift6759-aiadlp', job_type="report") as run:
			#Test Loss and Accuracy #Eval 1
			test_loss, test_accuracy = self.test(test_loader)
			run.summary.update({"test/loss": test_loss, "test/accuracy": test_accuracy})
			
			#Per Class Accuracy
			correct_pred, total_pred = self.per_class_accuracy(test_loader) 

			#Temporary Print of dictionary TODO: Move into WANDB #Eval 2
			for classname, correct_count in correct_pred.items():
				accuracy = 100. * correct_count / total_pred[classname]
				print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

			#K-hardest examples #Eval 3
			testset = self.tensor_dataset(test_loader)
			highest_losses, hardest_examples, true_labels, predictions = self.get_hardest_k_examples(testset)
			wandb.log({"high-loss-examples":
			           [wandb.Image(hard_example, caption= "Pred: " + str(int(pred)) + ", Label: " +  str(int(label)))
			            for hard_example, pred, label in zip(hardest_examples, predictions, true_labels)]})
			
			#Confusion Matrix #Eval 4
			#TODO
	
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

	@torch.no_grad():
	def get_hardest_k_examples(self, dataset, k=32):
		"""
		Finds the K-Hardest Examples fromt the given dataset
		"""
		loader = DataLoader(dataset, batch_size=1, shuffle=False)

		losses = None
		predictions = None

		self.eval()
		for data, targets in loader:
			data, targets = data.to(self.device), targets.to(self.device)
			outputs = self.forward(data)
			loss = self.criterion(outputs, targets)
			pred = outputs.argmax(dim=1, keepdim=True)

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
		predictions = predictions[argsort_loss[-k:]]

		return highest_k_losses, hardest_k_examples, true_labels, predictions

	def tensor_dataset(self, dataloader):
		"""
		Utility function which converts given DataLoader's Dataset into TensorDataset
		"""
		x, y = torch.tensor(dataloader.dataset.data), torch.tensor(dataloader.dataset.targets)
		return TensorDataset(x, y)