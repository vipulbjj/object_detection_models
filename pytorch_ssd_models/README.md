# Single Shot MultiBox Foot Detector Implementation in Pytorch
This repo implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325). The design goal is modularity and extensibility.

Currently, it has MobileNetV1, MobileNetV2, and VGG based SSD/SSD-Lite implementations. 

## Dependencies
1. Python 3.6+
2. OpenCV
3. Pytorch 1.0 or Pytorch 0.4+
4. Caffe2
5. Pandas

## Pretrained Models

### Mobilenet V1 SSD

URL: https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth

### MobileNetV2 SSD-Lite

URL: https://storage.googleapis.com/models-hao/mb2-ssd-lite-mp-0_686.pth

### VGG SSD

URL: https://storage.googleapis.com/models-hao/vgg16-ssd-mp-0_7726.pth

## Training

```bash
bash train.sh
```

## Evaluation

```bash
bash eval.sh 
```
You can freeze the base net, or all the layers except the prediction heads. 

```
  --freeze_base_net     Freeze base net layers.
  --freeze_net          Freeze all the layers except the prediction head.
```

You can also use different learning rates 
for the base net, the extra layers and the prediction heads.

```
  --lr LR, --learning-rate LR
  --base_net_lr BASE_NET_LR
                        initial learning rate for base net.
  --extra_layers_lr EXTRA_LAYERS_LR
```

As subsets of data can be very unbalanced, it also provides
a handy option to roughly balance the data.

```
  --balance_data        Balance training data by down-sampling more frequent
                        labels.
```

### Test on images

#### Testing on entire test folder

```bash
bash testing_200_examples.sh
```

#### Testing on a single test image

```bash
bash run_example.sh
```

##### Remember to edit the relevant bash files accordingly before running

## TODO

Insights on improving foot detection - https://slack-files.com/TN9K4QQNL-F0136RRJ1DK-11e69bd52d
