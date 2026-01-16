from ultralytics import YOLO
import cv2
import os
import numpy as np

# -------------------- CONFIG --------------------
MODEL_PATH = "E:/Project_Work/2025/Saffron_Project/Github_Code/Weed_Detection/Weed_Detection_Using_AI/segment/train/weights/best.pt"
INPUT_FOLDER = "Dataset/test/images"
OUTPUT_FOLDER = "results_batch"

# CONF_THRESH = 0.4
# IOU_THRESH = 0.5

CONF_THRESH = 0.3
IOU_THRESH = 0.4

# Supported image formats
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def batch_predict():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load model
    model = YOLO(MODEL_PATH)
    class_names = model.names

    image_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(IMG_EXT)
    ]

    print(f"\nTotal images found: {len(image_files)}\n")

    for img_name in image_files:
        img_path = os.path.join(INPUT_FOLDER, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Could not read image: {img_name}")
            continue

        # Run inference
        results = model(
            img_path,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            device=0
        )

        print(f"\n========== Image: {img_name} ==========")

        for r in results:
            boxes = r.boxes
            masks = r.masks

            if boxes is None:
                print(" No detections")
                continue

            for i in range(len(boxes)):
                # ---- Class & Confidence ----
                cls_id = int(boxes.cls[i])
                cls_name = class_names[cls_id]
                conf = float(boxes.conf[i])

                # ---- Bounding Box ----
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                print(f" Object {i+1}")
                print(f"  Class ID   : {cls_id}")
                print(f"  Class Name : {cls_name}")
                print(f"  Confidence : {conf:.3f}")
                print(f"  BBox       : [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

                # ---- Segmentation Polygon ----
                if masks is not None:
                    polygon = masks.xy[i]
                    print(f"  Polygon Points ({len(polygon)}):")

                    for p in polygon:
                        print(f"    (x={int(p[0])}, y={int(p[1])})")

                    # Draw polygon
                    poly_np = np.array(polygon, dtype=np.int32)
                    cv2.polylines(img, [poly_np], True, (0, 255, 0), 2)

                # Draw bounding box + label
                cv2.rectangle(
                    img,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 0, 0),
                    2
                )

                label = f"{cls_name} {conf:.2f}"
                cv2.putText(
                    img,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

        # Save result
        save_path = os.path.join(OUTPUT_FOLDER, img_name)
        cv2.imwrite(save_path, img)

        print(f" Saved: {save_path}")

    print("\nBatch prediction completed successfully")


if __name__ == "__main__":
    batch_predict()
