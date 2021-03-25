# Homework 2
Amanda Buddemeyer, CS 2770 Computer Vision, Spring 2021

## Running This Code
Each section of the assignment has its own script (hw2\_a.py, hw2\_b.py, hw2\_c.py).  All scripts use argparse, so use the --help flag to get a description of command-line arguments.

The corresponding Jupyter shows typical ways to run the scripts from the command line.  In some cases where noted, results reflect running the script with different parameters.

## Dependencies
|Package           | Version |
|------------------ |-------|
|click              |7.1.2|
|cycler             |0.10.0|
|Cython             |0.29.22|
|greenlet           |1.0.0|
|importlib-metadata |3.7.3|
|itsdangerous       |1.1.0|
|joblib             |1.0.1|
|kiwisolver         |1.3.1|
|Mako               |1.1.4|
|MarkupSafe         |1.1.1|
|matplotlib         |3.3.4|
|numpy              |1.20.1|
|Pillow             |8.1.2|
|pip                |21.0.1|
|pycocotools        |2.0.2|
|pyparsing          |2.4.7|
|python-dateutil    |2.8.1|
|python-editor      |1.0.4|
|scikit-learn       |0.24.1|
|scipy              |1.6.1|
|setuptools         |40.8.0|
|six                |1.15.0|
|sklearn            |0.0|
|threadpoolctl      |2.1.0|
|torch              |1.8.0|
|torchvision        |0.9.0|
|typing-extensions  |3.7.4.3|
|zipp               |3.4.1|

## Part A Results
####Accuracy:
0.4958
####Confusion Matrix:
||aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|aeroplane|22|0|0|1|0|0|0|1|0|0|1|0|0|0|0|0|0|0|0|0|
|bicycle|0|11|1|0|0|0|2|0|1|0|1|1|1|1|5|1|0|0|0|0|
|bird|0|0|20|1|0|0|0|1|2|0|0|0|0|0|0|0|1|0|0|0|
|boat|1|1|1|18|0|0|1|0|1|0|0|0|0|0|1|0|0|0|1|0|
|bottle|1|0|0|0|10|0|1|0|3|0|2|1|0|0|1|3|0|2|0|1|
|bus|1|0|0|0|0|15|2|0|0|0|1|1|0|0|0|0|0|0|1|0|
|car|1|1|0|1|0|0|11|0|2|0|1|0|0|0|5|2|0|1|0|0|
|cat|0|0|2|0|1|0|1|14|0|0|0|2|1|1|1|0|0|0|0|2|
|chair|0|0|1|1|0|0|1|1|7|0|2|0|1|0|4|3|0|2|0|2|
|cow|1|0|1|0|0|2|0|0|1|3|0|1|3|0|0|0|1|1|1|0|
|diningtable|0|0|0|0|4|0|0|1|2|1|5|0|0|1|1|6|0|1|1|2|
|dog|1|0|2|2|0|0|0|1|1|0|0|11|2|0|1|2|1|1|0|0|
|horse|0|1|0|0|0|0|0|0|0|2|1|1|14|0|1|2|0|1|0|0|
|motorbike|0|1|0|0|1|0|3|0|0|0|0|0|0|19|0|1|0|0|0|0|
|person|1|1|1|0|3|0|2|0|2|0|1|1|2|1|5|2|0|2|0|1|
|pottedplant|0|4|0|1|3|0|0|0|3|0|4|1|0|0|2|4|0|1|0|2|
|sheep|0|1|0|1|0|1|1|0|1|1|0|1|0|0|0|0|9|1|0|0|
|sofa|0|0|0|0|2|0|0|0|5|0|1|2|0|0|3|2|0|7|0|3|
|train|0|0|0|2|0|2|3|0|0|0|0|0|0|0|0|1|0|0|17|0|
|tvmonitor|0|0|1|0|2|0|0|0|2|0|2|0|0|0|1|0|0|3|0|14|

## Part B Results
###Learning rate 0.001, Batch Size 8, Optimizer optim.SGD
#### Accuracy
0.8447
#### Confusion Matrix
||aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|aeroplane|197|0|0|1|0|0|2|0|0|0|0|0|0|0|0|0|0|0|0|0|
|bicycle|0|178|0|0|2|3|4|0|1|0|0|0|0|2|4|3|0|0|2|1|
|bird|0|0|193|1|0|0|1|0|1|0|0|1|0|0|1|0|1|0|0|1|
|boat|1|0|0|192|0|0|3|0|0|0|0|0|0|0|2|1|0|1|0|0|
|bottle|0|5|0|0|132|0|2|4|8|0|13|6|2|0|5|5|1|7|0|10|
|bus|1|2|0|0|0|161|4|0|0|0|0|0|0|1|0|1|0|0|1|0|
|car|4|5|0|1|1|13|161|1|0|1|0|3|0|4|5|1|0|0|0|0|
|cat|0|0|0|0|1|0|1|191|0|0|0|3|0|0|2|1|0|1|0|0|
|chair|0|0|0|0|3|0|2|6|115|0|21|2|0|0|5|15|0|9|0|22|
|cow|0|1|0|0|0|0|0|0|0|120|0|0|0|0|0|0|0|0|0|0|
|diningtable|0|0|0|0|15|0|0|2|17|0|143|0|0|0|0|14|0|6|0|3|
|dog|0|0|1|0|0|0|2|4|2|0|1|175|1|0|4|2|2|5|0|1|
|horse|0|0|0|0|0|0|0|0|0|0|0|1|184|1|3|1|0|0|0|0|
|motorbike|0|4|0|1|0|3|5|0|0|0|0|0|0|175|10|1|1|0|0|0|
|person|0|2|0|4|4|2|14|0|8|0|4|1|7|4|143|2|0|3|2|0|
|pottedplant|0|2|1|0|2|0|2|5|8|0|17|3|0|3|1|136|0|11|4|5|
|sheep|0|0|0|0|0|0|0|1|0|1|0|0|0|0|0|0|135|0|0|0|
|sofa|0|1|0|0|0|0|0|5|4|0|2|5|0|0|2|19|0|154|0|8|
|train|0|0|0|0|0|0|4|0|1|0|0|0|0|0|0|0|0|0|195|0|
|tvmonitor|0|0|0|0|7|0|0|4|8|0|2|0|0|0|4|13|0|16|0|146|

### Learning Rate 0.0001, Batch Size 8, Optimizer optim.SGD
This model used a smaller learning rate than the previous, but the effect seems almost negligible on the accuracy.  The accuracy of this model is a bit lower than the previous model, though that could just be noise.  Sometimes making the learning rate smaller when using SGD can improve accuracy because larger learning rates can "step over" a loss minimum.  Sometimes making the learning rate larger will improve accuracy because the SGD algorithm will take larger "steps" to get closer to the loss minimum.  Neither seems to be the case here, which may just indicate that the new learning rate is not sufficiently different from the old one to make a difference.

#### Accuracy
0.8395
#### Confusion Matrix
||aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|aeroplane|193|0|0|2|0|1|4|0|0|0|0|0|0|0|0|0|0|0|0|0|
|bicycle|0|181|0|0|0|6|3|0|0|1|0|0|0|4|1|1|0|1|2|0|
|bird|0|0|193|1|0|0|1|0|1|0|0|1|0|0|1|1|1|0|0|0|
|boat|0|1|0|193|0|0|2|0|0|0|0|1|0|1|1|0|0|1|0|0|
|bottle|0|7|0|0|129|0|2|3|6|0|16|6|2|0|4|3|1|5|0|16|
|bus|0|1|0|0|0|161|5|0|0|0|0|0|0|1|1|0|0|0|2|0|
|car|2|6|0|2|1|13|159|1|0|1|0|3|0|5|3|1|0|0|3|0|
|cat|0|0|0|0|1|0|1|181|3|0|0|5|0|0|1|2|1|2|0|3|
|chair|0|1|0|0|4|0|1|4|113|0|28|5|0|0|5|6|0|10|1|22|
|cow|0|0|0|0|0|0|0|0|0|121|0|0|0|0|0|0|0|0|0|0|
|diningtable|0|0|0|0|14|0|0|0|16|0|153|0|0|0|2|8|0|6|0|1|
|dog|0|0|1|0|0|0|2|2|2|0|1|182|1|0|0|2|2|5|0|0|
|horse|0|0|0|0|0|0|0|0|0|0|0|1|187|1|1|0|0|0|0|0|
|motorbike|0|0|0|0|0|4|4|0|0|0|0|0|0|187|2|2|1|0|0|0|
|person|0|6|1|4|10|1|16|0|9|1|1|4|9|10|118|1|0|3|1|5|
|pottedplant|0|4|0|1|1|0|2|4|14|0|24|3|1|2|1|106|0|22|4|11|
|sheep|0|0|0|0|0|0|0|0|0|1|0|0|0|0|0|0|136|0|0|0|
|sofa|0|0|0|0|2|0|0|2|9|0|6|6|0|1|1|6|0|153|0|14|
|train|0|0|0|0|0|0|2|0|0|0|0|0|0|0|1|0|0|0|197|0|
|tvmonitor|0|0|1|0|2|0|0|2|11|0|2|1|0|0|0|3|0|15|0|163|

### Learning Rate 0.001, Batch Size 16, Optimizer optim.SGD
This model uses a larger batch size than the previous two.  While the larger batch size will make the training process slower and use more memory, it also increases the likelihood of finding a global loss minimum.  The accuracy of this model is exactly the same as the first model, though.  This implies either that the smaller batch size was finding the optimum weights and the larger batch size was not necessary, or that the batch size of 16 wasn't sufficiently larger than the batch size of 8 to find the minimum.

#### Accuracy
0.8447

#### Confusion Matrix
||aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|aeroplane|197|0|0|1|0|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0|
|bicycle|0|177|0|1|4|7|5|0|1|0|0|0|0|2|2|0|0|0|0|1|
|bird|0|0|194|1|0|0|1|0|1|0|0|0|0|0|0|1|1|0|0|1|
|boat|1|0|0|192|0|0|4|0|0|0|0|0|0|0|2|0|0|1|0|0|
|bottle|0|3|0|0|144|0|3|4|7|0|18|2|0|0|6|2|0|2|0|9|
|bus|0|0|0|0|0|167|4|0|0|0|0|0|0|0|0|0|0|0|0|0|
|car|4|3|0|0|1|15|162|0|0|0|0|5|0|3|5|1|0|0|1|0|
|cat|0|0|0|0|0|0|1|183|3|0|2|3|0|0|2|2|1|1|0|2|
|chair|0|0|0|0|5|0|2|2|117|0|29|3|0|0|7|11|0|5|1|18|
|cow|0|1|0|0|0|0|1|0|0|118|0|0|0|0|0|0|1|0|0|0|
|diningtable|0|0|0|0|12|0|0|0|11|0|158|1|0|0|4|8|0|3|0|3|
|dog|0|0|2|0|5|0|1|4|1|0|0|176|0|0|3|1|2|3|0|2|
|horse|0|0|0|0|2|0|0|0|0|0|0|2|179|1|6|0|0|0|0|0|
|motorbike|0|2|0|1|0|5|5|0|0|0|0|0|0|184|3|0|0|0|0|0|
|person|0|3|1|4|5|2|15|0|6|0|0|1|4|10|143|0|0|0|1|5|
|pottedplant|0|3|0|1|2|2|2|2|11|0|25|3|1|4|2|118|0|13|2|9|
|sheep|0|0|0|0|1|0|0|0|0|0|0|0|0|1|0|0|135|0|0|0|
|sofa|0|0|0|0|6|0|0|4|9|0|6|8|0|1|3|14|0|133|0|16|
|train|0|2|0|0|0|1|3|0|0|0|0|0|0|0|1|1|0|0|192|0|
|tvmonitor|0|0|0|0|8|0|0|2|13|0|4|0|0|0|1|7|0|8|0|157|

### Learning Rate 0.001, Batch Size 8, Optimizer optim.Adam
For this model, I switched the optimizer to the Adam algorithm.  I admit that I don't know much about the Adam algorithm, just that it's a replacement for SGD.  I tried to follow PyTorch's documentation for using this algorithm, but clearly I did something wrong because the algorithm just decided that everything was a chair.
#### Accuracy
0.0524

#### Confusion Matrix
||aeroplane|bicycle|bird|boat|bottle|bus|car|cat|chair|cow|diningtable|dog|horse|motorbike|person|pottedplant|sheep|sofa|train|tvmonitor|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|aeroplane|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|bicycle|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|bird|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|boat|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|bottle|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|bus|0|0|0|0|0|0|0|0|171|0|0|0|0|0|0|0|0|0|0|0|
|car|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|cat|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|chair|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|cow|0|0|0|0|0|0|0|0|121|0|0|0|0|0|0|0|0|0|0|0|
|diningtable|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|dog|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|horse|0|0|0|0|0|0|0|0|190|0|0|0|0|0|0|0|0|0|0|0|
|motorbike|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|person|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|pottedplant|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|sheep|0|0|0|0|0|0|0|0|137|0|0|0|0|0|0|0|0|0|0|0|
|sofa|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|train|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|
|tvmonitor|0|0|0|0|0|0|0|0|200|0|0|0|0|0|0|0|0|0|0|0|

