import cv2 as cv
from methods.color.color import get_dominent_colors


cap = cv.VideoCapture('../video_resources/kamchatka/kamchatka_shot_15.mp4')

count = 1
while True:
    _, img = cap.read()
    if img is None:
        break
    if count % 10 == 0:
        cv.imwrite(str(count) + '.jpg', img)
        get_dominent_colors(img, if_show=True, name='./vis_' + str(count))
    count += 1
