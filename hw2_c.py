import argparse, copy, os, pathlib, utils
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pascal_dataset import PASCALDataset
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import torch.optim as optim
from torch.optim import lr_scheduler
from PennFudanDataset import PennFudanDataset

# Get command-line arguments
parser = argparse.ArgumentParser(description='CS2770 HW2 Part C')
parser.add_argument('--epochs', type=int, default=25, help='The number of epochs')
parser.add_argument('--pascal_dir', type=pathlib.Path, required=True, help='Path to the PASCAL data set')
parser.add_argument('--penn_fudan_dir', type=pathlib.Path, required=True, help='Path to the PennFudan data set')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='The threshold to use for mAP calculation')
parser.add_argument('--output', nargs='?', type=argparse.FileType('w'), default='-', help='The output file where results will go')
args = parser.parse_args()

dataset_funcs = {
	'PASCAL': lambda x: PASCALDataset(os.path.join(args.pascal_dir, x)),
	'PF': lambda x: PennFudanDataset(os.path.join(args.penn_fudan_dir, x))
}

num_classes = {
	'PASCAL': 6,
	'PF': 2
}

device = 'cuda:0' if torch.cuda.is_available() else "cpu"
print(f'Device is {device}')

def get_iou(bb1, bb2):

	area1 = abs(bb1[0] - bb1[2]) * abs(bb1[1] - bb1[3])
	area2 = abs(bb2[0] - bb2[2]) * abs(bb2[1] - bb2[3])
			
	x_overlap = max(0.0, min(abs(bbox1[0] - bbox2[2]), abs(bbox1[2] - bbox2[0])))
	y_overlap = max(0.0, min(abs(bbox1[1] - bbox2[3]), abs(bbox1[3] - bbox2[1])))
	intersection_area = x_overlap * y_overlap
	
	union_area = area1 + area2 - intersection_area
	
	return float(intersection_area)/float(union_area)
	
def map_score_for_class(pred_bbs, gt_bbs):

	if gt_bbs is None or len(gt_bbs) == 0:
		return 0.0
	
	tps = 0

	for pred_bb in pred_bbs:
		isTP = False
		for gt_bb in gt_bbs:
			iou = get_iou(pred_bb, gt_bb)
			if iou >= args.iou_threshold:
				isTP = True
				break
		
		tps = tps + 1 if isTP else tps
	
	return float(tps)/float(len(pred_bbs))

def map_score(dataset, pred_bbs, gt_bbs):

	map_sum = 0
	for c in range(num_classes[dataset]):
		map_sum += map_score_for_class(pred_bbs[c], gt_bbs[c])
	
	return float(map_sum)/float(num_classes[dataset])

def get_test_results(model, dataloader):

	model.eval()
	coco = get_coco_api_from_dataset(data_loader.dataset)
	iou_types = ["bbox"]
	coco_evaluator = CocoEvaluator(coco, iou_types)
	
	for images, targets in dataloader:
		images = [image.to(device) for image in images]
		outputs = model(images).to('cpu')
		res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
		coco_evaluator.update(res)
	
	coco_evaluator.synchronize_between_processes()
	coco_evaluator.accumulate()
	coco_evaluator.summarize()
	
	print("Eval:")
	print(coco_evaluator.coco_eval['bbox'].stats[0])
		
# Create and train the model
def make_model(dataset):

	image_datasets = {x: dataset_funcs[dataset](x) for x in ['train', 'val', 'test']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn) for x in ['train', 'val' , 'test']}
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	num_ftrs = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(num_ftrs, num_classes[dataset])
	model = model.to(device)
	
	optimizer = optim.SGD(model.parameters(), lr=.001, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	
	best_model_wts = copy.deepcopy(model.state_dict())
	best_map = 0.0
	
	for epoch in range(args.epochs):
		print(f'Part C {dataset} Epoch {epoch+1} out of {args.epochs}: Training')
		
		model.train()

		for images, targets in dataloaders['train']:
			images = [image.to(device) for image in images]
			targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
			optimizer.zero_grad()
			loss_dict = model(images, targets)
			loss = sum(loss_dict.values())
			loss.backward()
			optimizer.step()
		
		scheduler.step()

		print(f'Part C {dataset} Epoch {epoch+1} out of {args.epochs}: Validation')
		get_test_results(model, dataloaders['val'])
		
	

[make_model(x) for x in ['PASCAL', 'PF']]





