import requests
import os
import cv2

def draw_line(img_folder, img_name, x_val):
    img_path = os.path.join(img_folder, "pred" + img_name)
    img = cv2.imread(os.path.join(img_folder, img_name))
    cv2.line(img, pt1=(x_val, 0), pt2=(x_val, 360), color=(255, 0, 0), thickness=2)
    print(img_path)
    cv2.imwrite(img_path, img)
def sendImg(img_path):
    KERAS_REST_API_URL = "http://127.0.0.1:7000/predict"
    image = open(img_path, "rb").read()
    payload = {"image": image}
    r = requests.post(KERAS_REST_API_URL, files=payload).json()
    if r["success"]:
        print(r)
        return r['predictions']
# 存放测试图片的文件夹
test_img_folder = "testImg"

img_name = "0.png"
# 生成路径
img_path = os.path.join(test_img_folder, img_name)
# 预测的pre值在0-1之间，即图片的滑块的中心坐标
pre = sendImg(img_path)

# 乘以360原因是原始图片大小为 140x360 图片宽为360
draw_line(test_img_folder, img_name, int(pre*360))
print("{} x坐标预测：{}".format(img_name, pre))
