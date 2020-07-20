from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import pandas as pd
import numpy as np
import torch
import glob

if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
image_path = sys.argv[4]

class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
    

    
image_files = glob.glob("data/test_100/*")
for count,image_path in enumerate(image_files):
    

    orig_image = cv2.imread(image_path)
    ###trying to add gt box too ##specific to foot_images
    image_name = image_path.split('/')[-1]
    dataset_type = image_path.split('/')[-2]
    string2 = dataset_type + "/" + image_name
#     df=pd.read_csv(image_path.replace(string2,"sub-"+ dataset_type + "-annotations-bbox.csv"))
#     gt_image=df[df['ImageID']==image_name]
#     gt_boxes=[]
#     gt_labels=[]
#     for i in range(gt_image.shape[0]):
#     #     print(i,df.shape)
#         curr_box=gt_image.iloc[i]
#         gt_boxes.append(torch.Tensor([curr_box['XMin'],curr_box['YMin'],curr_box['XMax'],curr_box['YMax']]))
#         gt_labels.append(curr_box['ClassName'])

    # print(len(gt_boxes),len(gt_labels))

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 10, 0.6)#confidence threshold reduced to 0.2
    # print(boxes)

    for i in range(boxes.size(0)):
        box = boxes[i, :]
#         gt_box = gt_boxes[i]
    #     print(type(box[0]),type(gt_box[0]))

        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
#         cv2.rectangle(orig_image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, 255), 4)
        #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
#         gt_label = "gt:" + gt_labels[i]

        cv2.putText(orig_image, label,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 255, 0),
                    2)  # line type

#         cv2.putText(orig_image, gt_label,
#                     (gt_box[0]+20, gt_box[1]-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,  # font scale
#                     (255, 0, 255),
#                     2)  # line type
        
#     for i in range(len(gt_boxes)):
#         gt_box = gt_boxes[i]
#         cv2.rectangle(orig_image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (255, 0, 255), 4)
#         gt_label = "gt:" + gt_labels[i]
#         cv2.putText(orig_image, gt_label,
#                     (gt_box[0]+20, gt_box[1]-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,  # font scale
#                     (255, 0, 255),
#                     2)  # line type
        
    path = "output_examples/test_100_latest_model/run_ssd_example_output_" +str(count) + ".jpg"
    cv2.imwrite(path, orig_image)
    print(f"Found {len(probs)} objects. The output image is {path}")