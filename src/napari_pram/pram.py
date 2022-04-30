import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndi

def pram_detect(img, thr_ctrss = 2):
    img = gaussian_filter(img, 3)               # filtering with Gaussian kernel/median filter 
    img = ndi.median_filter(img, 3, mode="constant")
    # Dynamic thresholding
    base_thr =  gaussian_filter(img, 31)
    # base_thr = ndi.median_filter(img, 31, mode="constant")
    tmp = np.zeros_like(img)
    for s in range(20):
        r_thr = base_thr - s
        cur   = np.where(img < r_thr, 1, 0).astype(np.uint8)
        kernel = np.ones((3,3),np.uint8)
        cur = cv2.erode(cur,kernel)
        # save_figure(cur, "logs/%d.png" % s)
        tmp+= cur
    # tmp = tmp/tmp.max()
    max_wnd = ndi.maximum_filter(tmp, size=15, mode='constant')
    tmp = np.where(tmp > thr_ctrss, tmp, 0)
    
    # Perform Non-maximum suppression
    tmp = np.where(max_wnd == tmp, tmp, 0)
    img=tmp
    preds = []
    # Find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find contours' center
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            preds.append([cX, cY])
        else:
            preds.append(c[0][0])
    preds = np.array(preds)
    return preds