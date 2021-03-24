import os
from scipy.io import loadmat

bbox_dir = "./PASCAL/train/BBox"
bbox_files = list(sorted(os.listdir(bbox_dir)))


for bbox_path in bbox_files:
	bbox = loadmat(os.path.join(bbox_dir, bbox_path))
	bbox = bbox['bboxes'].flatten()
	
	left_label = "left side first" if bbox[0] < bbox[2] else "right side first"
	top_label = "top first" if bbox[1] < bbox[3] else "bottom first"
	print(f'{left_label}, {top_label}')		