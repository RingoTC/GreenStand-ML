from pycocotools.coco import COCO
from PIL import Image
import os
import numpy as np
import cv2
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Process COCO dataset")
    parser.add_argument("--parent-folder", required=True, help="Parent folder containing train and test folders")
    parser.add_argument("--output-folder", required=True, help="Output folder for images, masks, and annotation.json")
    return parser.parse_args()

def main():
    args = parse_args()

    annotation_data = {"train": {}, "test": {}}

    for subset in ["train", "test"]:
        annotation_file = os.path.join(args.parent_folder, subset, "_annotations.coco.json")  # 修改为实际的文件名
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        os.makedirs(os.path.join(args.output_folder, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(args.output_folder, subset, "masks"), exist_ok=True)

        coco = COCO(annotation_file)

        image_ids = coco.getImgIds()

        for index, image_id in enumerate(image_ids):
            image_info = coco.loadImgs(image_id)[0]
            image_path = os.path.join(args.parent_folder, subset, image_info['file_name'])
            image = Image.open(image_path)

            output_image_path = os.path.join(args.output_folder, subset, "images", f"{index:04d}.jpg")
            image.save(output_image_path)

            annotation_ids = coco.getAnnIds(imgIds=image_id)

            masks = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

            bbox_list = []
            for ann_id in annotation_ids:
                annotation_info = coco.loadAnns(ann_id)[0]

                masks += coco.annToMask(annotation_info)

                bbox = annotation_info["bbox"]
                bbox_list.append(bbox)

            masks[masks == 1] = 255

            output_mask_path = os.path.join(args.output_folder, subset, "masks", f"{index:04d}_mask.jpg")
            cv2.imwrite(output_mask_path, masks)

            image_filename = f"{index:04d}.jpg"
            annotation_data[subset][image_filename] = {
                "mask_path": output_mask_path,
                "bbox_list": bbox_list
            }

    with open(os.path.join(args.output_folder, "annotation.json"), "w") as json_file:
        json.dump(annotation_data, json_file, indent=2)

if __name__ == "__main__":
    main()
