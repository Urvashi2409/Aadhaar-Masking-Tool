# py -u .\1_aadhar_mask_ocr.py --image images\img7.jpg to run the code

import pytesseract
import argparse
import re
from scipy import ndimage
from matplotlib import image
from matplotlib import pyplot as plt
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# To detect orientation and perform rotation
def rotate(image, center = None, scale = 1.0):
    angle=int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    rotated = ndimage.rotate(image, float(angle) * 1)
    return rotated

# To perform preprocessing before OCR
def preprocessing(image):
    w, h = image.shape[0],image.shape[1]
    if w < h:
        image = rotate(image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return threshold_img, image

# To perform OCR and mask first 8 digits
def aadhar_mask_and_ocr(thres_image, resized_image):
    d = pytesseract.image_to_data(thres_image, output_type=pytesseract.Output.DICT)
    keys = list(d.keys())
    n_boxes = len(d['text'])
    index = -1
    for i in range(0, n_boxes - 2):
        if len(d['text'][i]) == len(d['text'][i+1]) and (d['text'][i].isnumeric()) and (d['text'][i+1].isnumeric()) and (d['text'][i+2].isnumeric()) and len(d['text'][i+1]) == len(d['text'][i+2]) and (len(d['text'][i]) == 4):
            index1 = i
            index2 = i + 1
            break
    
    text = pytesseract.image_to_string(resized_image)
    print(text)
    
    left1 = d['left'][index1]
    top1 = d['top'][index1]
    h1 = d['height'][index1]
    w1 = d['width'][index1]
    digits1 = d['text'][index1]


    left2 = d['left'][index2]
    top2 = d['top'][index2]
    h2 = d['height'][index2]
    w2 = d['width'][index2]
    digits2 = d['text'][index2]
  
    final_image1 = cv2.rectangle(resized_image, (left1, top1), ( left1+w1, top1+h1), (255, 255, 255), -1)
    final_image = cv2.rectangle(final_image1, (left2, top2), ( left2+w2, top2+h2), (255, 255, 255), -1)

    final_image = cv2.resize(final_image, None, fx=0.33, fy=0.33)
    return final_image

# Command-line argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image to be scanned")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
image = rotate(image)
thres_image, resized_image = preprocessing(image)
masked_image = aadhar_mask_and_ocr(thres_image, resized_image)

print('Masked digits in given image. Displaying...')
cv2.imshow('mask' + args["image"], masked_image)

# To save ouput, uncomment below line
cv2.imwrite('mask' + args["image"], masked_image)
print('Press q over output window to close')
cv2.waitKey(0)