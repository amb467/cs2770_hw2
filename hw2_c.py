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

	bb1 = [min(bb1[0], bb1[2]), min(bb1[1], bb1[3]), max(bb1[0], bb1[2]), max(bb1[1], bb1[3])]
	bb2 = [min(bb2[0], bb2[2]), min(bb2[1], bb2[3]), max(bb2[0], bb2[2]), max(bb2[1], bb2[3])]
	
	area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
	area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
	
	x_overlap = max(0.0, min(abs(bb1[0] - bb2[2]), abs(bb1[2] - bb2[0])))
	y_overlap = max(0.0, min(abs(bb1[1] - bb2[3]), abs(bb1[3] - bb2[1])))
	intersection_area = x_overlap * y_overlap
	union_area = area1 + area2 - intersection_area
	return float(intersection_area)/float(union_area)
	
# {'boxes': tensor([[ 31.,  19., 461., 474.]]), 'labels': tensor([3]), 'image_id': tensor([86]), 'area': tensor([195650.]), 'iscrowd': tensor([0])}
def map_score(dataset, pred_bbs, gt_bbs):

	tp = np.zeros(num_classes[dataset]).tolist()
	fp = np.zeros(num_classes[dataset]).tolist()
	
	for pred_bb_item, gt_bb_item in zip(pred_bbs, gt_bbs):
		pred_boxes = pred_bb_item['boxes'].tolist()
		pred_labels = pred_bb_item['labels'].tolist()
		gt_boxes = gt_bb_item['boxes'].tolist()
		gt_labels = gt_bb_item['labels'].tolist()
		
		gts_by_label = defaultdict(list)
		for bbox, label in zip(gt_boxes, gt_labels):
			gts_by_label[label].append(bbox)
		
		for pred_bbox, label in zip(pred_boxes, pred_labels):
			if label in gts_by_label:
				isTP = False
				for gt_bbox in gts_by_label[label]:
					iou = get_iou(pred_bbox, gt_bbox)
					if iou >= args.iou_threshold:
						isTP = True
						tp[label] += 1
						break
				
				fp[label] += 0 if isTP else 1
					
			else:
				fp[label] += 1	
	
	maps = []
	for c in range(num_classes[dataset]):
		total = float(tp[c] + fp[c])
		
		if total > 0.0:
			maps.append(float(tp[c])/total)
	
	return sum(maps)/float(len(maps))

def get_test_results(model, dataloader):

	model.eval()
	coco = get_coco_api_from_dataset(dataloader.dataset)
	iou_types = ["bbox"]
	coco_evaluator = CocoEvaluator(coco, iou_types)
	all_targets = []
	
	for images, targets in dataloader:
		images = [image.to(device) for image in images]
		outputs = [{k: v.to(device) for k, v in t.items()} for t in model(images)]
		res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
		coco_evaluator.update(res)
		all_targets.extend(list(targets))
	
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





