ROI_ALIAS_CENTER = 0
ROI_ALIAS_LEFT = 1
ROI_ALIAS_RIGHT = 2


import cv2 as cv
import numpy as np
import os

from numba import jit, njit

# @jit(forceobj=True)
def OptimizeImage(img: cv.Mat) -> cv.Mat:
    if len(img.shape) == 3:
        for i in range(3):
            img[:, :, i] = cv.equalizeHist(img[:, :, i])
    else:
        img = cv.equalizeHist(img)
    img = cv.GaussianBlur(img, (5, 5), 0)
    return img

# @jit(nopython=True)
def SelectROI(img: cv.Mat, ROI_scale: float, alias: int = 0) -> cv.Mat:
    '''
        img: image to select ROI
        ROI_scale: scale of ROI
        alias: alias of ROI -> {0: center, 1: left, 2: right}
    '''
    if len(img.shape) == 3:
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return None
    h, w = img.shape
    roi_h = int(h * ROI_scale)
    roi_w = int(w * ROI_scale)
    roi_x = int((w - roi_w) / 2)
    roi_y = int((h - roi_h) / 2)
    if alias == 0:
        return img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    elif alias == 1:
        return img[roi_y:roi_y+roi_h, :roi_w]
    elif alias == 2:
        return img[roi_y:roi_y+roi_h, -roi_w:]
    else:
        return None


def GetImageDescriptor(sift: cv.SIFT, image_directory: str) -> dict[str: np.ndarray]:
    '''
        sift: cv.SIFT_create()
        image_directory: directory of image
    '''
    filenames = os.listdir(image_directory)
    # create an empty dictionary to store descriptors
    des_dict = {}
    # get descriptor of all images with jpg format
    for filename in filenames:
        if filename.endswith(".jpg"):
            img = cv.imread(os.path.join(image_directory, filename), cv.IMREAD_GRAYSCALE)
            img = OptimizeImage(img)
            _, des = sift.detectAndCompute(img, None)
            key_name = filename.split(".")[0]
            des_dict[key_name] = des
    return des_dict if len(des_dict) > 0 else None


def MergeDescriptors(des_dict: dict[str: np.ndarray], threshold: int = 20) -> dict[str: np.ndarray]:
    '''
    @brief: merge similar descriptors
        des_dict: list of descriptors of local frames -> dict[name: str, des: list]
    '''
    keys = list(des_dict.keys())
    length_keys = len(keys)
    bf = cv.BFMatcher()
    new_des_dict = {}
    for i in range(length_keys):
        for j in range(i+1, length_keys):
            if keys[j] in new_des_dict.keys():
                continue
            des1 = des_dict[keys[i]]
            des2 = des_dict[keys[j]]
            matches = bf.knnMatch(des1, des2, k=2)
            good_match = 0
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_match += 1
            if good_match > threshold:
                new_des_dict[keys[i]] = np.concatenate((des1, des2), axis=0)
                print(f"{keys[i]} and {keys[j]} merged.") if __name__ == '__main__' else None
                break
    return new_des_dict

# @jit(forceobj=True) 
def KNNMatch(bf: cv.BFMatcher, current_des: np.ndarray, local_des_dict: dict) -> dict[str: int]:
    '''
        current_des: descriptor of current frame
        local_des_dict: list of descriptors of local frames -> list[dict[name: str, des: list]]
    '''
    match_list = {}
    for key in local_des_dict.keys():
        des = local_des_dict[key]
        matches = bf.knnMatch(current_des, des, k=2)
        good_match = 0
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_match += 1
        match_list[key] = good_match

    print(match_list) if __name__ == '__main__' else None # Debug use. It proves that optimized image performs better.
    # return the best match
    return {max(match_list, key=match_list.get): match_list[max(match_list, key=match_list.get)]}


            

if __name__ == '__main__':
    img = cv.imread("faces/face_11.jpg")
    img = OptimizeImage(img)
    cv.imshow("img", img)
    cv.waitKey(0)