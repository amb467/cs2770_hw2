import argparse, pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='CS2770 HW2')
parser.add_argument('--parts', type=str, default='A,B', help='A comma-delimited list of the homework parts to process')
parser.add_argument('--data_dir', type=pathlib.Path, help='The data set to use for training, testing, and validation')
parser.add_argument('--output', nargs='?', type=argparse.FileType('w'), default='-', help='The output file where results will go')
args = parser.parse_args()
parts = args.parts.split(',')

device = 'cuda:0' if torch.cuda.is_available() else "cpu"
print(f'Device is {device}')

data_transforms = {
	'train': transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'val': transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'test': transforms.Compose([
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

class VGG16_Feature_Extraction(torch.nn.Module):
	def __init__(self):
		super(VGG16_Feature_Extraction, self).__init__()
		VGG16_Pretrained = models.vgg16(pretrained=True)
		self.features = VGG16_Pretrained.features
		self.avgpool = VGG16_Pretrained.avgpool
		self.feature_extractor = nn.Sequential(*[VGG16_Pretrained.classifier[i] for i in range(6)])
		
	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.feature_extractor(x)
		return x

def part_A():

	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=4) for x in ['train', 'val' , 'test']}

	model = VGG16_Feature_Extraction()
	model = model.to(device)

	print('Part A: Extracting Features')
	image_features = {}
	image_labels = {}
	for phase in ['train', 'test']:
		for inputs, labels in dataloaders[phase]:
			print(f'\tExtracting features for inputs {inputs} and labels {labels}')
			inputs = inputs.to(device)
			model_prediction = model(inputs)
			model_prediction_numpy = model_prediction.cpu().detach().numpy()
			if (phase not in image_features):
				image_features[phase] = model_prediction_numpy
				image_labels[phase] = labels.numpy()
			else:
				image_features[phase] = np.concatenate((image_features[phase], model_prediction_numpy), axis=0)
				image_labels[phase] = np.concatenate((image_labels[phase], labels.numpy()), axis=0)	

	print('Part A: Training SVM')
	clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0))
	clf.fit(image_features['train'], image_labels['train'])

	print('Part A: Testing SVM')

	return image_labels['test'], clf.predict(image_features['test'])
	
def part_B(learning_rate, batch_size, optimizer):

	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val' , 'test']}

	model = models.vgg16(pretrained=True)
	model = model.to(device)

	print('Part B: Training and Validating CNN')
	num_ftrs = model.classifier[6].in_features
	model.classifier[6] = nn.Linear(num_ftrs, len(class_names))

	num_epochs = 25
	criterion = nn.CrossEntropyLoss()
	optimizer = optimizer(model.parameters(), lr=learning_rate, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print(f'Epoch {epoch} out of {num_epochs}')
		for phase in ['train', 'test']:
			if phase == 'train':
				model.train()
			else:
				model.eval()

			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)

				optimizer.zero_grad()
		
				with torch.set_grad_enabled(phase == 'train'):
					inputs - inputs.to('cuda:0')
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)
					if phase == 'train':
						loss.backward()
						optimizer.step()
				all_batchs_loss += loss.item() * inputs.size(0)
				all_batchs_corrects += torch.sum(preds == labels.data)
		
				if phase == 'train':
					scheduler.step()
			
				epoch_loss = all_batchs_loss / dataset_sizes[phase]
				epoch_acc = all_batchs_corrects.double() / dataset_sizes[phase]
		
				if phase == 'val' and epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(model.state_dict())
					torch.save(best_model_wts , 'best_model_weight.pth')
				
	print('Part B: Testing CNN')
	model = models.vgg16()
	num_ftrs = model.classifier[6].in_features
	model.classifier[6] = nn.Linear(num_ftrs, 20)
	model = model.to(device)
	model.load_state_dict(torch.load('best_model_weight.pth'))
	model.eval()
	phase = 'test'
	
	y_true = {}
	y_pred = {}
	for inputs, labels in dataloaders[phase]:
		inputs = inputs.to(device)
		labels = labels.to(device)
		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)
		all_batchs_corrects += torch.sum(preds == labels.data)
		epoch_acc = all_batchs_corrects.double() / dataset_sizes[phase]
		y_true = np.concatenate((y_true, labels,to('cpu')), axis=0)
		y_pred = np.concatenate((y_pred, preds.to('cpu')), axis=0)	
		
	return y_true, y_pred

def print_results(part, y_true, y_pred):
	args.output.write(f"{part} Accuracy Score: {accuracy_score(y_true, y_pred)}")
	args.output.write(f"{part} Confusion Matrix")
	args.output.write(confusion_matrix(y_true, y_pred))
	

if 'A' in parts:
	print('Part A')
	y_true, y_pred = part_A()
	print_results('Part A', y_true, y_pred)

if 'B' in parts:

	for learning_rate in [0.001]:
		for batch_size in [8]:
			for optimizer in [optim.SGD]:
				part = f'Part B - learning rate {learning_rate}, batch_size {batch_size}, optimizer {optimizer}'
				y_true, y_pred = part_B(learning_rate, batch_size, optimizer)
				#print_results(part, y_true, y_pred)