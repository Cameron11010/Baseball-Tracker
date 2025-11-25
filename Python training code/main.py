import cv2
import os

# Paths
dataset_folder = '/labelling/images.cv/data/'  # Root folder of dataset
splits = ['train', 'val', 'test']
class_id = 0  # Baseball class

# Globals
drawing = False
ix, iy = -1, -1
current_boxes = []

def draw_box(event, x, y, flags, param):
    global ix, iy, drawing, img, current_boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            for (bx1, by1, bx2, by2) in current_boxes:
                cv2.rectangle(img_copy, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_boxes.append((ix, iy, x, y))
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow('image', img)


# Loop through splits
for split in splits:
    img_folder = os.path.join(dataset_folder, split, 'baseball')
    labels_folder = os.path.join(dataset_folder, split, 'labels')
    os.makedirs(labels_folder, exist_ok=True)

    img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        current_boxes = []

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_box)

        print(f"[{split}] Labeling {img_name}. Press 's' to preview/save, 'n' to skip, 'r' to reset boxes.")

        while True:
            cv2.imshow('image', img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                # Preview step
                preview_img = img.copy()
                for (x1, y1, x2, y2) in current_boxes:
                    cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imshow('Preview', preview_img)
                print("Previewing boxes. Press 'y' to save, 'n' to cancel.")
                while True:
                    k = cv2.waitKey(0) & 0xFF
                    if k == ord('y'):
                        label_path = os.path.join(labels_folder, os.path.splitext(img_name)[0] + '.txt')
                        with open(label_path, 'w') as f:
                            for (x1, y1, x2, y2) in current_boxes:
                                x_center = ((x1 + x2) / 2) / w
                                y_center = ((y1 + y2) / 2) / h
                                width = abs(x2 - x1) / w
                                height = abs(y2 - y1) / h
                                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        cv2.destroyWindow('Preview')
                        break
                    elif k == ord('n'):
                        print("Canceled saving boxes for this image.")
                        cv2.destroyWindow('Preview')
                        break
                break

            elif key == ord('n'):
                break
            elif key == ord('r'):
                current_boxes = []
                img = cv2.imread(img_path)

cv2.destroyAllWindows()
print("All YOLO annotations saved successfully with preview!")
