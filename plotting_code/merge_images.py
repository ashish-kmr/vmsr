import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def dilate_mask(img):
    kernel = np.ones((5,5), np.uint8)
    return cv2.dilate(img, kernel, iterations=2)

def dilate_mask_max(m):
    idx = np.where(m == 255)
    if len(idx[0]) == 0:
        return m
    minx = np.min(idx[0]); maxx = np.max(idx[0])
    miny = np.min(idx[1]); maxy = np.max(idx[1])
    m[minx:maxx, miny:maxy] = 255
    return m

#base = 'process/expt00/run04/'
for dirpath, dirnames, filenames in os.walk('process'):
    if dirpath[-5:-2] == 'run':
        base = dirpath
        #base = 'process/expt00/run04/'
        cap = cv2.VideoCapture(os.path.join(base,'third_person_video.mp4'))
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

        frame_list = []
        cnt = 0
        bg_frame = None
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is None: break
            if bg_frame is None: bg_frame = 1.0*frame
            else: bg_frame+=1.0*frame
            if cnt%120 == 0:
                frame_list.append(frame)
            cnt+=1
            #if cnt == 5000: break

        bg_frame = np.array(bg_frame*1.0/cnt, 'uint8')

        fgbg.apply(bg_frame)
        mask_list = [fgbg.apply(frm) for frm in frame_list]
        mask_list = [dilate_mask(frm) for frm in mask_list]

        frame_list = np.array(frame_list)
        mask_list = [np.repeat(np.expand_dims(frm,2),3, axis=2)/255 for frm in mask_list]

        sum_img = frame_list[0]#bg_frame
        alpha = 1.0
        for i in range(len(frame_list)-1, -1, -1):
            removed_img = mask_list[i]*sum_img
            sum_img = (1 - mask_list[i])*sum_img + alpha * mask_list[i]*frame_list[i] + (1.0 - alpha) * removed_img
            #alpha -= 0.1
            #alpha = max(alpha, 0.5)

        sum_mask = np.sum(mask_list, 0)

        cv2.imwrite(os.path.join(base,'merged_img.png'), sum_img)
        cv2.imwrite(os.path.join(base,'merged_mask.png'), 255*sum_mask)
        cv2.imwrite(os.path.join(base,'bg.png'), bg_frame)
        cv2.imwrite('process/merged_images/' + '_'.join(base.split('/')) + 'merged.png', sum_img)

        cap.release()
        cv2.destroyAllWindows()
