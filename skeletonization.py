import cv2
import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm
from inference.models.utils import get_roboflow_model
import supervision as sv

ROBOFLOW_API_KEY = 'epFQopoVFFvoH77vYrJT'
SOURCE_VIDEO_PATH = './Dataset/7tools_yolov8/test_video/cropped_video_12.mp4'
TARGET_VIDEO_PATH = './Dataset/7tools_yolov8/test_video/'
COLORS = sv.ColorPalette.DEFAULT
mask_annotator = sv.MaskAnnotator(color=COLORS)
# Load the video
video_path = 'your_video_path.mp4'
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
model = get_roboflow_model(model_id="sugical-tool-iszjm/1", api_key=ROBOFLOW_API_KEY)

# Iterate through each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)
    # Convert the frame to grayscale
    extracted_mask = detections.mask
    mask = detections.mask[0]
    # gray_frame = cv2.cvtColor(extracted_mask, cv2.COLOR_BGR2GRAY)

    # Perform skeletonization
    # skeleton = skeletonize(extracted_mask).astype(np.uint8)
    # for mask in extracted_mask:

    skeleton = skeletonize(mask)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    # skeleton_resized = cv2.resize(skeleton_uint8, (frame.shape[1], frame.shape[0]))
    # Convert the skeleton back to 3-channel format
    skeleton_rgb = cv2.cvtColor(skeleton_uint8, cv2.COLOR_GRAY2BGR)
    # print(skeleton_uint8.shape)

    overlaid_frame = cv2.addWeighted(frame, 1, skeleton_rgb, 0.5, 0)
    # Display the skeletonized frame
    # cv2.imshow('Skeletonized Frame', skeleton_uint8)

    annotated_frame = mask_annotator.annotate(
            scene=overlaid_frame, detections=detections
        )

    cv2.imshow('Overlaid Frame', annotated_frame)

    # Check if the user pressed 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video object and close all windows
cap.release()
cv2.destroyAllWindows()
