# -*- coding: utf-8 -*-
import cv2, os, argparse, shutil
import threading
from datetime import datetime

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 6  # fps

    ORG_WINDOW_NAME = "org"

    DEVICE_ID = 0

    # 分類器の指定
    #face_cascade_file = "/usr/local/Cellar/opencv/3.3.1_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
    #smile_cascade_file = "/usr/local/Cellar/opencv/3.3.1_1/share/OpenCV/haarcascades/haarcascade_smile.xml"
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier("cascade.xml")

    # カメラ映像取得
    cap = cv2.VideoCapture(0)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(ORG_WINDOW_NAME)

    smile_flame = 0
    face_size = 0
    smile_size = 0
    count = 0

    # 変換処理ループ
    while end_flag == True:

        # 画像の取得と顔の検出
        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = face_cascade.detectMultiScale(img_gray, 1.3,3)

        # 検出した顔に印を付ける
        for (x, y, w, h) in face_list:
            face_size = w*h
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0),2)
            roi_gray = img_gray[y+int(h/2):y+h, x:x+w]
            roi_color = img[y+int(h/2):y+h, x:x+w]
            smile_list = smile_cascade.detectMultiScale(roi_gray,1.3,3)
            for (ex,ey,ew,eh) in smile_list:
                smile_size = ew*eh
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                if smile_size != 0:
                    print('{0},{1},{2}'.format(y,h,ey))
                    #誤検出排除
                    #if (ey > h /2):
                    print("smile")
                    smile_flame += 1
                    """else:
                        print("no")
                        smile_flame = 0
"""
                if smile_flame >= 1:
                    print("save")
                    now = datetime.now().strftime('%Y%m%d%H%M%S')
                    #認識結果の保存
                    image_path = './outputs/' + now + "_" + str(count) + '.jpg'
                    cv2.imwrite(image_path, c_frame)
                    count += 1
                    smile_flame = 0

     
        cv2.imshow(ORG_WINDOW_NAME, img)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
