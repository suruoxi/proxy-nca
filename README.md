## About 

A pytorch toy of ICCV paper No Fuss Distance Metric Learning using Proxies.  The proxynca.py is from https://github.com/dichotomies/proxy-nca.



## Usage

```python train.py  --data-path /path/to/imagefolder/ --workers 8 --classes num_classes --save-prefix 'demo' --batch-size 1024 --gpus 0,1,2,3,4,5,6,7 --use-pretrained --model resnet50 --exclude-positive```

