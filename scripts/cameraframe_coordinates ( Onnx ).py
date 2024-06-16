import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import json
import math
import os

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

class CustomYOLO(YOLO):
    def set_class_names(self, names):
        self._names = names

    @property
    def names(self):
        return self._names
x1=0
y1=0
def run_yolo():
    W = 640
    H = 480

    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    model = CustomYOLO(r"Final-model.onnx", task='detect')

    with open('class_names.json', 'r') as f:
        class_names = json.load(f)

    model.set_class_names(class_names)

    print("Class names:", model.names)

    min_z_coordinate = None

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

        results = model(color_image, conf=0.6)

        object_coordinates = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].to('cpu').detach().numpy().copy()
                c = int(box.cls)
                class_name = class_names.get(c, f"Unknown_{c}")  # Handle missing class names
                cv2.rectangle(depth_colormap, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                # cv2.putText(depth_colormap, text=class_name, org=(int(b[0]), int(b[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_4)
                x = (int(b[0]) + int(b[2])) / 2
                y = (int(b[1]) + int(b[3])) / 2
                dept = depth_image[int(y), int(x)] / 1000
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                object_pos = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dept)
                if object_pos[2] > 0:
                    object_coordinates.append(object_pos)

        if object_coordinates:
            min_z_coordinate = min(object_coordinates, key=lambda coord: coord[2])
            d=math.sqrt((x1-min_z_coordinate[0])**2+(y1-min_z_coordinate[1])**2)
            print("Coordinate with minimum z value: ", min_z_coordinate)
            print(d)
            break

        annotated_frame = results[0].plot()
        cv2.imshow("color_image", annotated_frame)
        cv2.imshow("depth_image", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo()
