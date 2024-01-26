# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import json
from detectron2.structures import BoxMode

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from centernet.config import add_centernet_config
# constants
WINDOW_NAME = "CenterNet2 detections"

from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def get_dior_dicts(img_dir, annotaion_dir):
    json_file = os.path.join(annotaion_dir, "NWPU10_annotations_val.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
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
        
    return dataset_dicts

def registrar_dataset():
    for d in ["val"]:

        DatasetCatalog.register("nwpu_" + d, lambda d=d: get_dior_dicts("/home/ubq3/bishalworkspace/Overhead_Imagery_survey/Datasets/NWPU VHR-10/positive image set", "/home/ubq3/bishalworkspace/Overhead_Imagery_survey/Datasets/NWPU VHR-10"))
        
        # MetadataCatalog.get("dior_" + d).set(thing_classes=['golffield', 'bridge', 'vehicle', 'Expressway-Service-area', 'harbor', 'ship', 
        # 'storagetank', 'baseballfield', 'chimney', 'groundtrackfield', 'trainstation', 'windmill', 
        # 'basketballcourt', 'tenniscourt', 'overpass', 'stadium', 'airplane', 'dam', 'airport', 'Expressway-toll-station'])
        MetadataCatalog.get("nwpu_" + d).set(thing_classes=['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 
        'ground track field', 'harbor', 'bridge', 'vehicle'])


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.INPUT.RANDOM_FLIP="none"
    cfg.DATASETS.TEST = ("nwpu_val", )
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    if cfg.MODEL.META_ARCHITECTURE in ['ProposalNetwork', 'CenterNetDetector']:
        cfg.MODEL.CENTERNET.INFERENCE_TH = args.confidence_threshold
        cfg.MODEL.CENTERNET.NMS_TH = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))


    registrar_dataset()

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    output_file = None

    print("#####Output Path: #########",args.output)
    print("The name of the dataset: ", cfg.DATASETS.TEST[0])
    print(cfg)

    if args.input:
        print("@@@@@@@@@@@@@@@@@@@@@@@@INPUT VALID@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            files = os.listdir(args.input[0])
            args.input = [args.input[0] + x for x in files]
            assert args.input, "The input path(s) was not found"
        visualizer = VideoVisualizer(
            MetadataCatalog.get(
                cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
            ), 
            instance_mode=ColorMode.IMAGE)
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            print("Image Path: ",path)
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(
                img, visualizer=visualizer)
            if 'instances' in predictions:
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )
            else:
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["proposals"]), time.time() - start_time
                    )
                )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                    visualized_output.save(out_filename)
                else:
                    # assert len(args.input) == 1, "Please specify a directory with args.output"
                    # out_filename = args.output
                    if output_file is None:
                        width = visualized_output.get_image().shape[1]
                        height = visualized_output.get_image().shape[0]
                        frames_per_second = 15
                        output_file = cv2.VideoWriter(
                            filename=args.output,
                            # some installation of opencv may not support x264 (due to its license),
                            # you can try other format (e.g. MPEG)
                            fourcc=cv2.VideoWriter_fourcc(*"x264"),
                            fps=float(frames_per_second),
                            frameSize=(width, height),
                            isColor=True,
                        )
                    output_file.write(visualized_output.get_image()[:, :, ::-1])
            else:
                # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                # cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(1 ) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = 15 # video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            # assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)

            cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            cv2.imshow(basename, vis_frame)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


























# # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import argparse
# import glob
# import multiprocessing as mp
# import os
# import time
# import cv2
# import tqdm
# import json
# from detectron2.data import MetadataCatalog, DatasetCatalog

# from detectron2.config import get_cfg
# from detectron2.data.detection_utils import read_image
# from detectron2.utils.logger import setup_logger
# from detectron2.structures import BoxMode

# from predictor import VisualizationDemo
# from centernet.config import add_centernet_config
# # constants
# WINDOW_NAME = "CenterNet2 detections"

# from detectron2.utils.video_visualizer import VideoVisualizer
# from detectron2.utils.visualizer import ColorMode, Visualizer
# from detectron2.data import MetadataCatalog

# def get_dior_dicts(img_dir, annotaion_dir):
#     json_file = os.path.join(annotaion_dir, "debug_anno.json")
#     with open(json_file) as f:
#         imgs_anns = json.load(f)

#     dataset_dicts = []
#     for idx, v in enumerate(imgs_anns.values()):
#         record = {}
        
#         filename = os.path.join(img_dir, v["filename"])
#         height, width = cv2.imread(filename).shape[:2]
        
#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = height
#         record["width"] = width

#         #print(filename)
#         annos = v["shape_attributes"]
#         objs = []
#         #print(annos)

#         for i,bbox in enumerate(annos):
#             #print(anno_list[0][str(0)][0][0])

#             xmin= bbox[str(i)][0][0]
#             ymax= bbox[str(i)][0][1]
#             width= bbox[str(i)][1][0]
#             height= bbox[str(i)][1][1]

#             obj = {
#                 "bbox": [xmin, ymax, int(width), int(height)],
#                 "bbox_mode": BoxMode.XYWH_ABS,
#                 #"segmentation": [poly],
#                 "category_id": int(bbox['category_id']),
#             }

#             objs.append(obj)
#         record["annotations"] = objs
#         dataset_dicts.append(record)
#         #print(dataset_dicts)
        
#     return dataset_dicts


# def registrar_dataset():
#     for d in ["val"]:

#         DatasetCatalog.register("dota_" + d, lambda d=d: get_dior_dicts("/home/ubq3/bishal workspace/Overhead_Imagery_survey/Datasets/DOTA2.0/debug1024/images", "/home/ubq3/bishal workspace/Overhead_Imagery_survey/Datasets/Datasets/DOTA2.0/debug1024"))
#         # MetadataCatalog.get("dior_" + d).set(thing_classes=['golffield', 'bridge', 'vehicle', 'Expressway-Service-area', 'harbor', 'ship', 'storagetank', 'baseballfield', 'chimney', 'groundtrackfield', 'trainstation', 'windmill', 'basketballcourt', 'tenniscourt', 'overpass', 'stadium', 'airplane', 'dam', 'airport', 'Expressway-toll-station'])
#         #MetadataCatalog.get("visdrone_" + d).set(thing_classes=["ignored regions", "pedestrian",   "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor", "others"])
#         MetadataCatalog.get("dota_" + d).set(thing_classes=['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                   'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'helipad', 'airport'])

# def setup_cfg(args):
#     # load config from file and command-line arguments
#     cfg = get_cfg()
#     add_centernet_config(cfg)
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
#     cfg.TEST.DETECTIONS_PER_IMAGE=256
#     # Set score_threshold for builtin models
#     cfg.DATASETS.TEST = ("dota_val", )
#     cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
#     if cfg.MODEL.META_ARCHITECTURE in ['ProposalNetwork', 'CenterNetDetector']:
#         cfg.MODEL.CENTERNET.INFERENCE_TH = args.confidence_threshold
#         cfg.MODEL.CENTERNET.NMS_TH = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
#     cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
#     cfg.freeze()
#     return cfg


# def get_parser():
#     parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
#     parser.add_argument(
#         "--config-file",
#         default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
#         metavar="FILE",
#         help="path to config file",
#     )
#     parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
#     parser.add_argument("--video-input", help="Path to video file.")
#     parser.add_argument("--input", nargs="+", help="A list of space separated input images")
#     parser.add_argument(
#         "--output",
#         help="A file or directory to save output visualizations. "
#         "If not given, will show output in an OpenCV window.",
#         default="/home/ubq3/bishalworkspace/Overhead_Imagery_survey/Datasets/DOTA2.0/debug1024/PRED/"
#     )

#     parser.add_argument(
#         "--confidence-threshold",
#         type=float,
#         default=0.5,
#         help="Minimum score for instance predictions to be shown",
#     )
#     parser.add_argument(
#         "--opts",
#         help="Modify config options using the command-line 'KEY VALUE' pairs",
#         default=[],
#         nargs=argparse.REMAINDER,
#     )
#     return parser


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     args = get_parser().parse_args()
#     logger = setup_logger()
#     logger.info("Arguments: " + str(args))

#     registrar_dataset()

#     cfg = setup_cfg(args)

#     demo = VisualizationDemo(cfg)
#     output_file = None
#     print("Configureation: \n",cfg)
#     print("Output Path: ",args.output)
#     print("The name of the dataset: ", cfg.DATASETS.TEST[0])

#     if args.input:
#         if len(args.input) == 1:
#             args.input = glob.glob(os.path.expanduser(args.input[0]))
#             files = os.listdir(args.input[0])
#             args.input = [args.input[0] + x for x in files]
#             assert args.input, "The input path(s) was not found"
#         visualizer = VideoVisualizer(
#             MetadataCatalog.get(
#                 cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
#             ), 
#             instance_mode=ColorMode.IMAGE)
#         for path in tqdm.tqdm(args.input, disable=not args.output):
#             # use PIL, to be consistent with evaluation
#             print("Image Path: ", path)
#             img = read_image(path, format="BGR")
#             start_time = time.time()
#             predictions, visualized_output = demo.run_on_image(
#                 img, visualizer=visualizer)
#             if 'instances' in predictions:
#                 logger.info(
#                     "{}: detected {} instances in {:.2f}s".format(
#                         path, len(predictions["instances"]), time.time() - start_time
#                     )
#                 )
#             else:
#                 logger.info(
#                     "{}: detected {} instances in {:.2f}s".format(
#                         path, len(predictions["proposals"]), time.time() - start_time
#                     )
#                 )

#             if args.output:
#                 print("I am here!!!!!!!!!!! : ", args.output)
#                 if os.path.isdir(args.output):
#                     assert os.path.isdir(args.output), args.output
#                     out_filename = os.path.join(args.output, os.path.basename(path))
#                     print("output filename : ",out_filename)
#                     visualized_output.save(out_filename)
#                 else:
#                     # assert len(args.input) == 1, "Please specify a directory with args.output"
#                     # out_filename = args.output
#                     if output_file is None:
#                         width = visualized_output.get_image().shape[1]
#                         height = visualized_output.get_image().shape[0]
#                         frames_per_second = 15
#                         output_file = cv2.VideoWriter(
#                             filename=args.output,
#                             # some installation of opencv may not support x264 (due to its license),
#                             # you can try other format (e.g. MPEG)
#                             fourcc=cv2.VideoWriter_fourcc(*"x264"),
#                             fps=float(frames_per_second),
#                             frameSize=(width, height),
#                             isColor=True,
#                         )
#                     output_file.write(visualized_output.get_image()[:, :, ::-1])
#             else:
#                 # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#                 #cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
#                 if cv2.waitKey(1 ) == 27000:
#                     break  # esc to quit
#     elif args.webcam:
#         assert args.input is None, "Cannot have both --input and --webcam!"
#         cam = cv2.VideoCapture(0)
#         for vis in tqdm.tqdm(demo.run_on_video(cam)):
#             #cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
#             #cv2.imshow(WINDOW_NAME, vis)
#             if cv2.waitKey(1) == 27:
#                 break  # esc to quit
#         cv2.destroyAllWindows()
#     elif args.video_input:
#         video = cv2.VideoCapture(args.video_input)
#         width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         frames_per_second = 15 # video.get(cv2.CAP_PROP_FPS)
#         num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#         basename = os.path.basename(args.video_input)

#         if args.output:
#             if os.path.isdir(args.output):
#                 output_fname = os.path.join(args.output, basename)
#                 output_fname = os.path.splitext(output_fname)[0] + ".mkv"
#             else:
#                 output_fname = args.output
#             # assert not os.path.isfile(output_fname), output_fname
#             output_file = cv2.VideoWriter(
#                 filename=output_fname,
#                 # some installation of opencv may not support x264 (due to its license),
#                 # you can try other format (e.g. MPEG)
#                 fourcc=cv2.VideoWriter_fourcc(*"x264"),
#                 fps=float(frames_per_second),
#                 frameSize=(width, height),
#                 isColor=True,
#             )
#         assert os.path.isfile(args.video_input)
#         for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
#             if args.output:
#                 output_file.write(vis_frame)

#             #cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
#             #cv2.imshow(basename, vis_frame)
#             if cv2.waitKey(1) == 27:
#                 break  # esc to quit
#         video.release()
#         if args.output:
#             output_file.release()
#         else:
#             pass
#             #cv2.destroyAllWindows()
