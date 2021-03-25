import argparse, copy, os, pathlib, utils, numpy as np
import torch
from pascal_dataset import PASCALDataset
from PennFudanDataset import PennFudanDataset
from collections import defaultdict

parser = argparse.ArgumentParser(description='CS2770 HW2 Part C')
parser.add_argument('--iou_threshold', type=float, default=0.5, help='The threshold to use for mAP calculation')
args = parser.parse_args()

num_classes = {
	'PASCAL': 6,
	'PF': 2
}

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

def format_targets(targets):
	new_targets = {}
	
	for target in targets:
		boxes = target['boxes'].tolist()
		labels = target['labels'].tolist()
		image_id = target['image_id'].tolist()[0]
		new_targets[image_id] = defaultdict(list)
		
		for bbox, label in zip(boxes, labels):
			new_targets[image_id][label].append(bbox)
		
	return new_targets

image_dataset = PASCALDataset('./PASCAL/test/')
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

gt_bbs = []
for images, targets in dataloader:
	#score = map_score('PASCAL', targets, targets)
	#print(f'Score: {score}')
	
	#for i in range(len(targets)):
	#	print(f'{i}:')
	#	print(targets[i])
	gt_bbs.extend(list(targets))
	
#gt_bbs = format_targets(epoch_targets)
score = map_score('PASCAL', gt_bbs, gt_bbs)
print(score)

		





