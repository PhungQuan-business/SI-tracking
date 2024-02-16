import argparse
import os
from typing import Dict, List, Set, Tuple
from tqdm import tqdm

import cv2
import numpy as np
from inference.models.utils import get_roboflow_model
import supervision as sv
from cv2 import KalmanFilter

SOURCE_VIDEO_PATH = './Dataset/7tools_yolov8/test_video/tool_video_11.mp4'
TARGET_VIDEO_PATH = './Dataset/7tools_yolov8/test_video/'
ROBOFLOW_API_KEY = 'epFQopoVFFvoH77vYrJT'
MODEL_ID = "sugical-tool-iszjm/1"
COLORS = sv.ColorPalette.DEFAULT

class VideoProcessor:
    def __init__(
            self,
            roboflow_api_key: str,
            model_id: str,
            source_video_path: str,
            source_target_path: str=None,
            confidence_threshold: float = 0.5,
            iou_threshold: float = 0.7,
            minute_start=None,
            minute_end = None,
            frame_rate=None
            ) -> None:
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.source_target_path = source_target_path

        self.model = get_roboflow_model(model_id=model_id, api_key=roboflow_api_key)
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.minute_start = minute_start
        self.minute_end = minute_end
        self.frame_rate = frame_rate
        
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.mask_annotator = sv.MaskAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator()
        
        # Chưa có Class này
        # self.detectionManager = detectionManager()

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path= self.source_video_path,
            start=60 * self.minute_start * self.frame_rate
            )
        for frame in frame_generator:
            processed_frame = self.process_frame(frame=frame)
            cv2.imshow("frame", processed_frame)

            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        
    # Output là frame đã được label và là đầu vào cho process_video()
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self.model.infer(
            frame, confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold
        )[0]
        detections = sv.Detections.from_inference(result)
        detection_with_mask = detections
        detections = self.tracker.update_with_detections(detections)
        return self.annotate_frame(frame=frame, detections=detections, detection_with_mask=detection_with_mask)
    
    def annotate_frame(
            self, 
            frame: np.ndarray, 
            detections: sv.Detections,
            detection_with_mask) -> np.ndarray:
        
        # Drawing bbox
        annotated_frame = frame.copy()


        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        # Drawing bbox
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        # Drawing mask
        annotated_frame = self.mask_annotator.annotate(
            scene=annotated_frame, detections=detection_with_mask
        )
        # Assign id for object
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels)
        
        return annotated_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=B"yteTrack for Surgical tools tracking"
    )
    parser.add_argument(
        "--model_id",
        default="ugical-tool-iszjm/1",
        help= "Roboflow model_id, is the name of pre-trained model",
        type=str
    )
    parser.add_argument(
        "--roboflow_api_key",
        default=None,
        help="Roboflow API key, can be take from user profile",
        type=str
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="path to the source video",
        type=str
    )
    parser.add_argument(
        "--source_target_path",
        default=None,
        help="output location for processed video",
        type=str
    )
    parser.add_argument(
        "--confidence_threshold",
        help='Confidence threshold for model',
        default=0.3,
        type=float
    )
    parser.add_argument(
        "--iou_threshold",
        help="iou threshold for model",
        default=0.7,
        type=float
    )
    parser.add_argument(
        "--minute_start",
        default=0,
        help="the point where the video start",
        type=int
    )
    parser.add_argument(
        "--frame_rate",
        default=24,
        help="frame rate",
        type=int
    )
    parser.add_argument(
        "--minute_end",
        default=0,
        help="the end minute position",
        type=int
    )

    #  Chưa hiểu cái này
    args =parser.parse_args()

    api_key = args.roboflow_api_key
    api_key = os.environ.get("ROBOFLOW_API_KEY", api_key)
    if api_key is None:
        raise ValueError(
            "Roboflow API KEY is missing. Please provide it as an argument or set the "
            "ROBOFLOW_API_KEY environment variable."
        )

    processor = VideoProcessor(
        roboflow_api_key=args.roboflow_api_key,
        model_id=args.model_id,
        source_video_path=args.source_video_path,
        source_target_path=args.source_target_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        minute_start=args.minute_start,
        minute_end=args.minute_end,
        frame_rate=args.frame_rate
    )
    processor.process_video()

'''
python ByteTrack.py \
--roboflow_api_key epFQopoVFFvoH77vYrJT \
--model_id sugical-tool-iszjm/1 \
--source_video_path ../Dataset/7tools_yolov8/test_video/cropped_video_12.mp4 \
--source_target_path ../Dataset/7tools_yolov8/test_video/ \
--minute_start 29 \
--minute_end 32 \
--frame_rate 20
# frame cua vid la 25 
'''

'''
Để có được tracker_id thì cần phải update_with_detection trước.
Lưu ý là khi update_with_detection xong thì mask sẽ bị mất(đã xử lí tạm thời được cái này)
'''
            

