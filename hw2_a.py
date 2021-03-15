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
#from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

data_dir = 'hw2_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=4) for x in ['train', 'val' , 'test']}
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
		

model = VGG16_Feature_Extraction()
device = 'cuda:0' if torch.cuda.is_available() else "cpu"
print(f'Device is {device}')
model = model.to(device)

print('Extracting features')
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

print('Training SVM')
clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0))
clf.fit(image_features['train'], image_labels['train'])

print('Testing SVM')

y_pred = clf.predict(image_features['test'])
confusion_matrix(image_labels['test'], y_pred)