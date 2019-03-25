## About 

A pytorch toy of ICCV paper No Fuss Distance Metric Learning using Proxies.  The proxynca.py is from https://github.com/dichotomies/proxy-nca.



## Prepare Dataset

[CUB 200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)  data is organized as 

```
cub_pytorch
|---train
|-----subdirectory 0000 to 0099
|---val
|-----subdirectory 0100 to 0199
```

## Train

see train_cub.sh



## Results


| Metric | [dichotomies's Implementation](https://github.com/dichotomies/proxy-nca)  | [Google's Implementation](https://arxiv.org/pdf/1703.07464.pdf) |  This Implementation(epoch 199) |
| ------ | -------------------- | ------------- | ----------------|
|  R@1   |       **52.46**      |     49.21     | 51.76 |
|  R@2   |       **64.78**      |     61.90     | 64.30 |
|  R@4   |       **75.38**      |     67.90     | 75.24 |
|  R@8   |       84.31      |     72.40     | **85.09** |
|  NMI   |       **60.84**      |     59.53     | - |


