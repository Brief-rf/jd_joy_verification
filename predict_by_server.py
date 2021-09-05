import requests
import os
import cv2
import time

def draw_line(img_folder, img_name, x_val):
    img_path = os.path.join(img_folder, "pred" + img_name)
    img = cv2.imread(os.path.join(img_folder, img_name))
    cv2.line(img, pt1=(x_val, 0), pt2=(x_val, 360), color=(255, 0, 0), thickness=2)
    cv2.imwrite(img_path, img)
def sendImg(img_path):
    KERAS_REST_API_URL = "http://127.0.0.1:7000/predict"
    image = open(img_path, "rb").read()
    payload = {"image": image}
    r = requests.post(KERAS_REST_API_URL, files=payload).json()
    if r["success"]:
        print(r)
        return r['predictions']
img_path = "testImg"
timeList = []
for testimg in os.listdir(img_path):
    img_p = os.path.join(img_path, testimg)
    start = time.time()
    pre = sendImg(img_p)
    cost_time = time.time()-start
    draw_line(img_path, testimg, int(pre*360))
    timeList.append(cost_time)
    print("{} x坐标预测：{} 耗时：{}s".format(testimg, pre, cost_time))

print("平均检测一张图片耗时：{}s".format(sum(timeList)/len(timeList)))