# Copyright 2023 Synopsys, Inc.
# This Synopsys software and all associated documentation are proprietary
# to Synopsys, Inc. and may only be used pursuant to the terms and conditions
# of a written license agreement with Synopsys, Inc.
# All other use, reproduction, modification, or distribution of the Synopsys
# software or the associated documentation is strictly prohibited.

from pathlib import Path
import cv2


def GetDataLoader(dataset=None, max_input=None, img_list=None, **kwargs):
    if img_list is None:
        extensions = [".jpg", ".jpeg", ".png"]
        path_to_all_files = sorted(Path(dataset).glob("*"))
        img_list = [image_path for image_path in path_to_all_files if image_path.suffix.lower() in extensions]
    for image_path in img_list[:max_input]:
        img = cv2.imread(str(image_path))
        img = preprocess(img)
        yield img


def resize(image, target_size=(512, 512)):
    aspect_ratio = image.shape[1] / image.shape[0]
    if aspect_ratio > 1:
        # Width is greater than height
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:
        # Height is greater than or equal to width
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)

    resized_image = cv2.resize(image, [new_width, new_height])

    pad_left = (target_size[0] - new_width) // 2
    pad_right = target_size[0] - new_width - pad_left
    pad_top = (target_size[1] - new_height) // 2
    pad_bottom = target_size[1] - new_height - pad_top
    padded_image = cv2.copyMakeBorder(resized_image,
                                      pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT,
                                      value=[255, 255, 255])
    return padded_image


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = resize(image)

    # normalize
    mean_vec = [123.6750, 116.2800, 103.5300]
    stddev_vec = [58.3950, 57.1200, 57.3750]
    norm_img_data = (image_resized - mean_vec) / stddev_vec

    # add batch channel
    norm_img_data = norm_img_data.transpose(2, 0, 1).reshape(1, 3, 512, 512).astype('float32')
    return norm_img_data
