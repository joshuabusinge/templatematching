from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import pytesseract
from pytesseract import Output

# x1,y1,x2,y2
health_unit_xy = [100,90,800,160]
level_xy = [850,90,1400,160]
district_xy = [100,170,800,230]
sub_county_xy = [850,170,1400,230]
period_xy = [100,240,800,300]
year_xy = [850,240,1400,300]
male_xy = [820,450,930,515]
female_xy = [1200,450,1310,515]
positive_xy = [820,525,930,590]
negative_xy = [1200,525,1310,590]
pf_xy = [820,600,930,665]
vivax_xy = [1200,600,1310,665]
age00 = [820,695,930,760]
age01 = [1200,695,1310,760]
age10 = [820,775,930,840]
age11 = [1200,775,1310,840]
age20 = [820,855,930,920]
age21 = [1200,855,1310,920]
age30 = [820,935,930,1000]
sym00 = [820,1050,930,1115]
sym01 = [1200,1050,1310,1115]
sym10 = [820,1140,930,1205]
sym11 = [1200,1140,1310,1205]
sym20 = [820,1220,930,1285]
sym21 = [1200,1220,1310,1285]

def draw_boxes(img):
    img = cv2.rectangle(img, (health_unit_xy[0], health_unit_xy[1]), (health_unit_xy[2], health_unit_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (level_xy[0], level_xy[1]), (level_xy[2], level_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (district_xy[0], district_xy[1]), (district_xy[2], district_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (sub_county_xy[0], sub_county_xy[1]), (sub_county_xy[2], sub_county_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (period_xy[0], period_xy[1]), (period_xy[2], period_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (year_xy[0], year_xy[1]), (year_xy[2], year_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (male_xy[0], male_xy[1]), (male_xy[2], male_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (female_xy[0], female_xy[1]), (female_xy[2], female_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (positive_xy[0], positive_xy[1]), (positive_xy[2], positive_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (negative_xy[0], negative_xy[1]), (negative_xy[2], negative_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (pf_xy[0], pf_xy[1]), (pf_xy[2], pf_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (vivax_xy[0], vivax_xy[1]), (vivax_xy[2], vivax_xy[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (age00[0], age00[1]), (age00[2], age00[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (age01[0], age01[1]), (age01[2], age01[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (age10[0], age10[1]), (age10[2], age10[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (age11[0], age11[1]), (age11[2], age11[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (age20[0], age20[1]), (age20[2], age20[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (age21[0], age21[1]), (age21[2], age21[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (age30[0], age30[1]), (age30[2], age30[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (sym00[0], sym00[1]), (sym00[2], sym00[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (sym01[0], sym01[1]), (sym01[2], sym01[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (sym10[0], sym10[1]), (sym10[2], sym10[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (sym11[0], sym11[1]), (sym11[2], sym11[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (sym20[0], sym20[1]), (sym20[2], sym20[3]), (255,0,0), 2)
    img = cv2.rectangle(img, (sym21[0], sym21[1]), (sym21[2], sym21[3]), (255,0,0), 2)

    return img

def show_steps(orig_img, contour_img, warped_img, boxes_img):

    h1, w1 = orig_img.shape[:2]
    h2, w2 = boxes_img.shape[:2]

    merged = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

    merged[:h1, :w1,:3] = orig_img
    merged[:h2, w1:w1+w2,:3] = boxes_img

    return merged

def arrange_points(pts):
    # Make a rectangle out of given points
    shape_rect = np.zeros((4, 2), dtype = "float32")
    shape_rect[0] = pts[np.argmin(pts.sum(axis = 1))]
    shape_rect[1] = pts[np.argmin(np.diff(pts, axis = 1))]
    shape_rect[2] = pts[np.argmax(pts.sum(axis = 1))]
    shape_rect[3] = pts[np.argmax(np.diff(pts, axis = 1))]
    return shape_rect

def transform_form(image, pts):
    rect = arrange_points(pts)
    (top_left, top_right, bottom_right, bottom_left) = rect
    widthA = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    widthB = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    heightB = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_form_edges(img):
    # convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 40, 100)
    return edged

def detect_form_contour(orig_img, edge_img):
    # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
    contours = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    # loop over the contours
    for c in contours:
        # approximate the contour
        approx_contour = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx_contour) == 4:
            rect = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            wh_ratio = rect[2]/rect[3] 
            if wh_ratio > 0.90 and wh_ratio < 1.10:
                formCnt = approx_contour
                break

    form_contour= cv2.drawContours(orig_img, [formCnt], -1, (0,255,0), 3)

    return formCnt, form_contour

def get_text_from_rect(img, xy):
    gray = cv2.cvtColor(img[xy[1]:xy[3], xy[0]:xy[2]], cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    return pytesseract.image_to_string(invert, lang='eng', config='--psm 6')

def get_radio_sum(radio):
    cv2.imwrite("radio.jpg", radio)
    radio = 255 - radio
    return radio.sum(0).sum(0)

def get_gender(img):
    male_radio_img = img[male_xy[1]:male_xy[3], male_xy[0]:male_xy[2]]
    female_radio_img = img[female_xy[1]:female_xy[3], female_xy[0]:female_xy[2]]
    if get_radio_sum(male_radio_img) > get_radio_sum(female_radio_img):
        return 'Male'
    else:
        return 'Female'

def get_diagnosis_result(img):
    pos_radio_img = img[positive_xy[1]:positive_xy[3], positive_xy[0]:positive_xy[2]]
    neg_radio_img = img[negative_xy[1]:negative_xy[3], negative_xy[0]:negative_xy[2]]
    if get_radio_sum(pos_radio_img) > get_radio_sum(neg_radio_img):
        return 'Positive'
    else:
        return 'Negative'

def get_malaria_species(img):
    pf_radio_img = img[pf_xy[1]:pf_xy[3], pf_xy[0]:pf_xy[2]]
    vivax_radio_img = img[vivax_xy[1]:vivax_xy[3], vivax_xy[0]:vivax_xy[2]]
    if get_radio_sum(pf_radio_img) > get_radio_sum(vivax_radio_img):
        return 'Pf'
    else:
        return 'Vivax'

def get_patient_age(img):
    age_radio_img_list = [
        img[age00[1]:age00[3], age00[0]:age00[2]]
        ,img[age01[1]:age01[3], age01[0]:age01[2]]
        ,img[age10[1]:age10[3], age10[0]:age10[2]]
        ,img[age11[1]:age11[3], age11[0]:age11[2]]
        ,img[age20[1]:age20[3], age20[0]:age20[2]]
        ,img[age21[1]:age21[3], age21[0]:age21[2]]
        ,img[age30[1]:age30[3], age30[0]:age30[2]]
    ]
    age_radio_sum_list = []
    for age_radio_img in age_radio_img_list:
        age_radio_sum_list.append(get_radio_sum(age_radio_img))

    max_age = age_radio_sum_list.index(max(age_radio_sum_list))

    if max_age == 0: return 'Below 5 yrs'
    elif max_age == 1: return '5-10 yrs'
    elif max_age == 2: return '11-20 yrs'
    elif max_age == 3: return '21-30 yrs'
    elif max_age == 4: return '31-40 yrs'
    elif max_age == 5: return '41-50 yrs'
    elif max_age == 6: return 'Above 51 yrs'
    else : return 'none'

def get_symptoms(img):
    symp_radio_img_list = [
        img[sym00[1]:sym00[3], sym00[0]:sym00[2]]
        ,img[sym01[1]:sym01[3], sym01[0]:sym01[2]]
        ,img[sym10[1]:sym10[3], sym10[0]:sym10[2]]
        ,img[sym11[1]:sym11[3], sym11[0]:sym11[2]]
        ,img[sym20[1]:sym20[3], sym20[0]:sym20[2]]
        ,img[sym21[1]:sym21[3], sym21[0]:sym21[2]]
    ]
    symp_radio_sum_list = []
    for symp_radio_img in symp_radio_img_list:
        symp_radio_sum_list.append(get_radio_sum(symp_radio_img))

    symptoms =[]

    sum_threshold = 15000

    if symp_radio_sum_list[0] > sum_threshold: symptoms.append('High fever')
    if symp_radio_sum_list[1] > sum_threshold: symptoms.append('Sweating')
    if symp_radio_sum_list[2] > sum_threshold: symptoms.append('Headache')
    if symp_radio_sum_list[3] > sum_threshold: symptoms.append('Vomiting')
    if symp_radio_sum_list[4] > sum_threshold: symptoms.append('Join Pain')
    if symp_radio_sum_list[5] > sum_threshold: symptoms.append('Diarrhea')

    return symptoms



def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
    args = vars(ap.parse_args())

    # read image
    orig_form = imutils.resize(cv2.imread(args["image"]), height = 1500)
    # orig_form = imutils.resize(cv2.imread("img1.jpg"), height = 1500)
    copy_form = orig_form.copy()

    # taking ratio of orginal image with resized image
    ratio = orig_form.shape[0] / 500.0

    # resizing the form image
    copy_form = imutils.resize(copy_form, height = 500)

    form_edges = detect_form_edges(copy_form.copy())
    #print("STEP 1: Edge Detection")

    formCnt, form_contour = detect_form_contour(copy_form.copy(), form_edges.copy())
    #print("STEP 2: Find contours of paper")
    
    form_warped = transform_form(orig_form.copy(), formCnt.reshape(4, 2) * ratio)
    form_warped = imutils.resize(form_warped, height = 1500)
    #print("STEP 3: Apply perspective transform")
    
    form_boxes = draw_boxes(form_warped.copy())
    #print("STEP 4: Draw Boxes")

    form_warped_thresh = cv2.cvtColor(form_warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(form_warped_thresh, 171, offset = 30, method = "gaussian")
    form_warped_thresh = (form_warped_thresh > T).astype("uint8") * 255
    #print("STEP 5: Adaptive Thresholding")


    health_unit = get_text_from_rect(form_warped.copy(), health_unit_xy)
    health_unit = health_unit.replace("HEALTH UNIT:", "").lstrip().rstrip()
    level = get_text_from_rect(form_warped.copy(), level_xy)
    level = level.replace("LEVEL:", "").lstrip().rstrip()
    district = get_text_from_rect(form_warped.copy(), district_xy)
    district = district.replace("DISTRICT:", "").lstrip().rstrip()
    sub_county = get_text_from_rect(form_warped.copy(), sub_county_xy)
    sub_county = sub_county.replace("SUBCOUNTY:", "").lstrip().rstrip()
    period = get_text_from_rect(form_warped.copy(), period_xy)
    period = period.replace("REPORTING PERIOD: MONTH:", "").lstrip().rstrip()
    year = get_text_from_rect(form_warped.copy(), year_xy)
    year = year.replace("YEAR:", "").lstrip().rstrip()
    gender = get_gender(form_warped_thresh.copy())
    diagnosis_result = get_diagnosis_result(form_warped_thresh.copy())
    if diagnosis_result == 'Negative':
        malaria_species = 'None'
    else :
        malaria_species = get_malaria_species(form_warped_thresh.copy())
    patient_age = get_patient_age(form_warped_thresh.copy())
    symptoms = get_symptoms(form_warped_thresh.copy())

    print('Health Unit :', health_unit)
    print('Level :', level)
    print('District :', district)
    print('Subcounty :', sub_county)
    print('Period :', period)
    print('Year :', year)
    print('Gender :', gender)
    print('Result :', diagnosis_result)
    print('Species :', malaria_species)
    print('Patient Age :', patient_age)
    print('Symptoms :', symptoms)



if __name__ == "__main__":
    main()





    

    