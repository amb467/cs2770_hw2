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
parser.add_argument('--lr', type=float, default=.001, help='The learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='The batch size')
parser.add_argument('--optimizer', type=str, default='SGD', help='The optimizer - SGD or Adam')
args = parser.parse_args()

optimizers = {
	'SGD': lambda params: optim.SGD(params, lr=args.lr, momentum=0.9),
	'Adam': lambda params: optim.Adam(params, lr=args.lr,)
}

if args.optimizer not in optimizers:
	print("Optimizer must be 'SGD' or 'Adam'")
	exit
	
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
	
def map_score(dataset, pred_bbs, gt_bbs):

	tp = np.zeros(num_classes[dataset]).tolist()
	fp = np.zeros(num_classes[dataset]).tolist()
	
	for gt_bb_item in gt_bbs:
		image_id = gt_bb_item['image_id'].item()
		pred_bb_item = pred_bbs[image_id]
		
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

def get_test_results(model, dataset, dataloader):

	model.eval()
	coco = get_coco_api_from_dataset(dataloader.dataset)
	iou_types = ["bbox"]
	coco_evaluator = CocoEvaluator(coco, iou_types)
	all_targets = []
	all_res = {}
	
	for images, targets in dataloader:
		images = [image.to(device) for image in images]
		outputs = [{k: v.to(device) for k, v in t.items()} for t in model(images)]
		res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
		coco_evaluator.update(res)
		all_targets.extend(list(targets))
		all_res.update(res)
	
	
	coco_evaluator.synchronize_between_processes()
	coco_evaluator.accumulate()
	coco_evaluator.summarize()
	
	coco_map_score = coco_evaluator.coco_eval['bbox'].stats[0]
	map_score = map_score(dataset, all_res, all_targets)
	print("Coco Evaluator: {coco_map_score}; my calculation: {map_score}")
	return map_score
		
# Create and train the model
def make_model(dataset):

	image_datasets = {x: dataset_funcs[dataset](x) for x in ['train', 'val', 'test']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=utils.collate_fn) for x in ['train', 'val' , 'test']}
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	num_ftrs = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(num_ftrs, num_classes[dataset])
	model = model.to(device)
	
	optimizer = optimizers[args.optimizer](model.parameters())
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
		epoch_map = get_test_results(model, dataset, dataloaders['val'])
		if epoch_map > best_map:
			best_map = epoch_map
			best_model_wts = copy.deepcopy(model.state_dict())
			torch.save(best_model_wts , 'part_c_best_model_weight.pth')
	
	print('Part C: {dataset} Testing')
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
	num_ftrs = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(num_ftrs, num_classes[dataset])
	model = model.to(device)
	model.load_state_dict(torch.load('part_c_best_model_weight.pth'))
	test_map = get_test_results(model, dataset, dataloaders['train'])
	print(f'Part C {dataset} mAP: {test_map}')
		
[make_model(x) for x in ['PASCAL', 'PF']]





