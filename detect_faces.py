from ultralytics import YOLO
from PIL import Image
import cv2
import os


def detect_faces_yolov8(image_path, model_path="yolov8n-face.pt", conf=0.3):
    model = YOLO(model_path)

    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    results = model(image_path, conf=conf)

    face_boxes = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            score = float(box.conf[0].cpu().numpy())

            cx = ((x1 + x2) / 2.0) / W
            cy = ((y1 + y2) / 2.0) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H

            face_boxes.append({
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "xywh_norm": [float(cx), float(cy), float(bw), float(bh)],
                "conf": score
            })

    return face_boxes


def draw_faces(image_path, face_boxes, save_path="output/detected_faces.jpg"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = cv2.imread(image_path)

    for face in face_boxes:
        x1, y1, x2, y2 = face["xyxy"]
        conf = face["conf"]

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"face {conf:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imwrite(save_path, img)
    print(f"偵測結果圖片已儲存到：{save_path}")


if __name__ == "__main__":
    image_path = "/home/albee/const_layout_whitespace/user_images/image01.png"
    model_path = "yolov8n-face.pt"

    face_boxes = detect_faces_yolov8(
        image_path=image_path,
        model_path=model_path,
        conf=0.3
    )

    print("偵測到的人臉數量:", len(face_boxes))

    for i, face in enumerate(face_boxes):
        print(f"\nFace {i + 1}")
        print("xyxy:", face["xyxy"])
        print("xywh_norm:", face["xywh_norm"])
        print("conf:", face["conf"])

    draw_faces(image_path, face_boxes)