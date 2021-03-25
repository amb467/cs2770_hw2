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
import numpy as np
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
parser.add_argument('--lr', type=float, default=.001, help='The learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='The batch size')
parser.add_argument('--optimizer', type=str, default='SGD', help='The optimizer - SGD or Adam')
args = parser.parse_args()

optimizers = {
	'SGD': lambda params: optim.SGD(params, lr=args.lr, momentum=0.9),
	'Adam': lambda params: optim.Adam(params, lr=args.lr,)
}

if args.optimizer not in optimizers:
	print("Optimizer must be 'SGD' or 'Adam'")
	exit
	
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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val' , 'test']}
class_names = image_datasets['train'].classes

def get_test_results(model, phase):

	model.eval()
	
	y_true = []
	y_pred = []
	
	for inputs, labels in dataloaders[phase]:
		inputs = inputs.to(device)
		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)
		y_true.extend(labels)
		y_pred.extend(preds.to('cpu'))
		
	return y_true, y_pred
	
print(f'Part B - learning rate {args.lr}, batch_size {args.batch_size}, optimizer {args.optimizer}')
args.output.write(f'Part B - learning rate {args.lr}, batch_size {args.batch_size}, optimizer {args.optimizer}\n')

model = models.vgg16(pretrained=True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optimizers[args.optimizer](model.parameters())
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(args.epochs):
	print(f'Epoch {epoch+1} out of {args.epochs}: Training')

	model.train()

	for inputs, labels in dataloaders['train']:
		inputs = inputs.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()

		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

	scheduler.step()

	print(f'Epoch {epoch+1} out of {args.epochs}: Validation')
	y_true, y_pred = get_test_results(model, 'val')
	epoch_acc = accuracy_score(y_true, y_pred)
	if epoch_acc > best_acc:
		best_acc = epoch_acc
		best_model_wts = copy.deepcopy(model.state_dict())
		torch.save(best_model_wts , 'part_b_best_model_weight.pth')

print('Testing')
model = models.vgg16()
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 20)
model = model.to(device)
model.load_state_dict(torch.load('part_b_best_model_weight.pth'))
y_true, y_pred = get_test_results(model, 'train')
args.output.write(f"Accuracy Score: {accuracy_score(y_true, y_pred)}\n")
args.output.write(f"Confusion Matrix\n")
args.output.write(np.array2string(confusion_matrix(y_true, y_pred)))