import os
from model import brief_net
import numpy as np
import cv2

def draw_line(img_folder, img_name, x_val):
    img_path = os.path.join(img_folder, "pred"+img_name)
    img = cv2.imread(os.path.join(img_folder, img_name))
    print(x_val)
    cv2.line(img, pt1=(x_val, 0), pt2=(x_val, 170), color=(0, 255, 255), thickness=2)
    cv2.imwrite(img_path, img)
from PIL import Image

img_path = "testImg"

if __name__ == '__main__':
    # 图片路径
    model = brief_net(input_shape=(140, 360, 3), output_shape=1)
    model.load_weights("trained_weights.h5")
    for testimg in os.listdir(img_path):
        img = Image.open(os.path.join(img_path, testimg))
        img = np.array(img) / 255.
        img = img[:,:,:3]
        print(img.shape)
        img = (np.expand_dims(img, 0))
        pre = model.predict(img)
        draw_line(img_path, testimg, int(pre*360))
    print("Done")