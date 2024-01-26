import cv2
import sys
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from matplotlib import pyplot as plt

from centernet.config import add_centernet_config

def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(), plt.imshow(im), plt.axis('off')


def get_visdrone_dicts(img_dir, annotaion_dir):
    json_file = os.path.join(annotaion_dir, "dota_commons_val.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    cnt=0
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        #print(filename)
        annos = v["shape_attributes"]
        objs = []
        #print(annos)

        for i,bbox in enumerate(annos):
            #print(anno_list[0][str(0)][0][0])

            xmin= bbox[str(i)][0][0]
            ymax= bbox[str(i)][0][1]
            width= bbox[str(i)][1][0]
            height= bbox[str(i)][1][1]

            obj = {
                "bbox": [xmin, ymax, int(width), int(height)],
                "bbox_mode": BoxMode.XYWH_ABS,
                #"segmentation": [poly],
                "category_id": int(bbox['category_id']),
            }

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        #print(dataset_dicts)

        # cnt+=1
        # if cnt == 200:
        #     return dataset_dicts
        
    return dataset_dicts

   

def registrar_dataset():
    for d in ["val"]:
        DatasetCatalog.register("dota_" + d, lambda d=d: get_visdrone_dicts("/home/ubq3/bishalworkspace/Overhead_Imagery_survey/Datasets/DOTA2.0/val1024/images_common", "/home/ubq3/bishalworkspace/Overhead_Imagery_survey/Datasets/DOTA2.0/val1024"))

        #MetadataCatalog.get("dota_" + d).set(thing_classes=['Small_Object','Large_Object'])

        # MetadataCatalog.get("dota_" + d).set(thing_classes=['small-vehicle','plane', 'Difficult_Object', 'large-vehicle', 'baseball-diamond', 'ship', 'harbor',
        #     'ground-track-field', 'storage-tank', 'basketball-court', 'tennis-court'])

        MetadataCatalog.get("dota_" + d).set(thing_classes=['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                    'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'helipad', 'airport'])


def start_validation():


    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    add_centernet_config(cfg)
    cfg.merge_from_file("/home/ubq3/SOD/CenterNet2/projects/CenterNet2/configs/CenterNet2-F_R50_1x.yaml")
    #cfg.DATASETS.TRAIN = ("visdrone_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 8
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.01 * 1 / 16  # pick a good LR
    #cfg.SOLVER.MAX_ITER = 144906    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR=cfg.OUTPUT_DIR+"/DOTA/Centernet2_CSP_COCO_trained_new"
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.WEIGHTS = "/home/ubq3/model_final.pth"  # path to the model we just trained

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    print(cfg)
    predictor = DefaultPredictor(cfg)

    #box_visualization(predictor)

    evaluator = COCOEvaluator("dota_val", output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "dota_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == "__main__":
    #argv = sys.argv
    #print (argv)
   
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    registrar_dataset()
    start_validation()
