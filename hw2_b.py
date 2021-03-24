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
parser.add_argument('--epochs', type=int, default=25, help='The number of epochs')
parser.add_argument('--data_dir', type=pathlib.Path, help='The data set to use for training, testing, and validation')
parser.add_argument('--output', nargs='?', type=argparse.FileType('w'), default='-', help='The output file where results will go')
args = parser.parse_args()

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
class_names = image_datasets['train'].classes

def get_test_results(model, phase):

	model.eval()
	
	y_true = []
	y_pred = []
	all_batchs_corrects = 0
	
	for inputs, labels in dataloaders[phase]:
		inputs = inputs.to(device)
		labels = labels.to(device)
		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)
		y_true.append(labels.to('cpu'))
		y_pred.append(preds.to('cpu'))
		
	return y_true, y_pred


for learning_rate in [0.0001, 0.001]:

	optimizers = {
		'SGD': lambda params: optim.SGD(params, lr=learning_rate, momentum=0.9),
		'Adam': lambda params: optim.Adam(params, lr=learning_rate)
	}
	
	for opt_label, optimizer in optimizers.items():
		for batch_size in [16, 8]:
		
			args.output.write(f'Part B - learning rate {learning_rate}, batch_size {batch_size}, optimizer {opt_label}')
			dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val' , 'test']}

			model = models.vgg16(pretrained=True)
			num_ftrs = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(num_ftrs, len(class_names))
			model = model.to(device)
	
			criterion = nn.CrossEntropyLoss()
			optimizer = optimizer(model.parameters())
			scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

			best_model_wts = copy.deepcopy(model.state_dict())
			best_acc = 0.0

			for epoch in range(args.epochs):
				print(f'Part B Epoch {epoch+1} out of {args.epochs}: Training')
		
				model.train()
		
				for inputs, labels in dataloaders['train']:
					inputs = inputs.to(device)
					labels = labels.to(device)
		
					optimizer.zero_grad()
			
					with torch.set_grad_enabled(phase == 'train'):
						outputs = model(inputs)
						_, preds = torch.max(outputs, 1)
						loss = criterion(outputs, labels)
						loss.backward()
						optimizer.step()

				scheduler.step()
		
				print(f'Part B Epoch {epoch+1} out of {args.epochs}: Validation')
				y_true, y_pred = get_test_results(model, 'val')
				epoch_acc = accuracy_score(y_true, y_pred)
				if epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(model.state_dict())
					torch.save(best_model_wts , 'part_b_best_model_weight.pth')
			
			print('Part B: Testing')
			model = models.vgg16()
			num_ftrs = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(num_ftrs, 20)
			model = model.to(device)
			model.load_state_dict(torch.load('part_b_best_model_weight.pth'))
			y_true, y_pred = get_test_results(model, 'train')
			args.output.write(f"Part B Accuracy Score: {accuracy_score(y_true, y_pred)}")
			args.output.write(f"Part B Confusion Matrix")
			args.output.write(confusion_matrix(y_true, y_pred))