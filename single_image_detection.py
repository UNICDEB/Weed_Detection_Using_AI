from ultralytics import YOLO
import cv2
import os
import numpy as np

def test_single_image():
    # -------------------- PATHS --------------------
    model_path = "E:/Project_Work/2025/Saffron_Project/Github_Code/Weed_Detection/Weed_Detection_Using_AI/segment/train/weights/best.pt"
    image_path = "demo.jpg"
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # -------------------- LOAD MODEL --------------------
    model = YOLO(model_path)

    # -------------------- RUN INFERENCE --------------------
    results = model(
        image_path,
        conf=0.3,
        iou=0.5,
        device=0
    )

    # -------------------- READ IMAGE --------------------
    img = cv2.imread(image_path)

    print("\n========== DETECTION RESULTS ==========\n")

    for r in results:
        boxes = r.boxes
        masks = r.masks
        names = model.names

        if boxes is None:
            print("No objects detected")
            return

        for i in range(len(boxes)):
            # -------- CLASS & CONFIDENCE --------
            cls_id = int(boxes.cls[i])
            cls_name = names[cls_id]
            conf = float(boxes.conf[i])

            # -------- BOUNDING BOX --------
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

            print(f"Object {i+1}")
            print(f" Class ID   : {cls_id}")
            print(f" Class Name : {cls_name}")
            print(f" Confidence : {conf:.3f}")
            print(f" BBox       : [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

            # -------- SEGMENTATION POLYGON --------
            if masks is not None:
                polygon = masks.xy[i]  # (N, 2)
                print(f" Polygon Points (Total {len(polygon)}):")

                for p in polygon:
                    print(f"   (x={int(p[0])}, y={int(p[1])})")

                # Draw polygon
                poly_np = np.array(polygon, dtype=np.int32)
                cv2.polylines(img, [poly_np], True, (0, 255, 0), 2)

            print("----------------------------------")

        # -------- SAVE RESULT IMAGE --------
        save_path = os.path.join(save_dir, "segmentation_result.jpg")
        cv2.imwrite(save_path, img)

        # -------- DISPLAY RESULT --------
        cv2.imshow("YOLO11x Segmentation Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"\nResult image saved at: {save_path}")

if __name__ == "__main__":
    test_single_image()
