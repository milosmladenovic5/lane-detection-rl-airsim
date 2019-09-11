import airsim
import numpy as np
import matplotlib.pyplot as pl
import cv2
import time
from scipy.spatial.distance import euclidean

client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False)

region_of_interest_vertices = [
    (0, 35),
    (0, 65),
    (255, 65),
    (35, 255),
    (205, 0)
]

MAX_DISTANCE_ALLOWED = 60
HEIGHT = 65
WIDTH  = 255
CENTER = np.array([WIDTH // 2, HEIGHT // 2])

GRAY_DIFFERENCE_THRESHOLD = 13
LEFT_RIGHT_DIFF_TRESHOLD = 15

LEFT_PIXEL_DISTANCE = 30
RIGHT_PIXEL_DISTANCE = 35

def region_of_interest(img, vertices):
        # Define a blank matrix that matches the image height/width.
        mask = np.zeros_like(img)    # Retrieve the number of color channels of the image.
        #channel_count = img.shape[2]    # Create a match color with the same color channel counts.
        match_mask_color = 255 # (255,) * channel_count - set this when image is not grayscale

        # Fill inside the polygon
        cv2.fillPoly(mask, vertices, match_mask_color)

        # Returning the image only where mask pixels match
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


def distance_from_the_line(line_start, line_end, point):
    """
    Computes the distance of a point from a given line
    http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    
    :param line_start: (np.array) - first point of the line 
    :param line_end: (np.array) - second point of the line 
    :param point: (np.array)
    :return: (float) point distance from a given line
    """
    return np.linalg.norm(np.cross(line_end - line_start, line_start - point)) /\
           np.linalg.norm(line_start - line_end)
          
def brightness(img, val):
    assert val > 0
#   uint8 overflow handling ...
#   e: 222 + 50 = 16
    limit = 255 - val
    img[img > limit] = 255
    img[img < limit] += val

def get_image(crop_h1=70, crop_h2=135, crop_w1=0, crop_w2=255):
    """
    Returns the cropped front camera image
    """
    image_response = client.simGetImages([airsim.ImageRequest("MyCamera1", airsim.ImageType.Scene, False, False)])[0]
    image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 3) # some envs. use diffrent number of channels (like l. mountains and nh)

    return image_rgba[crop_h1:crop_h2, crop_w1:crop_w2, 0:3]

def detect_lines(img):
    frame = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_yellow = np.array([18, 25, 140])
    up_yellow = np.array([48, 200, 160])
    mask = cv2.inRange(hsv, low_yellow, up_yellow)
    
    edges = cv2.Canny(mask, 100, 150)
    #cv2.imshow('frame',edges)
    
    #cropped_image = region_of_interest(edges, np.array([region_of_interest_vertices], np.int32))
    #lines = cv2.HoughLines(cropped_image, 2, np.pi/180, 25) #, maxLineGap=3)
    lines = cv2.HoughLinesP(edges, 2, np.pi/180, 25, maxLineGap=3)
    
    return lines

def val_diff_greater_than_threshold(threshold, first_val, second_val, third_val):
    diff_first_second = 0
    diff_second_third = 0
    diff_first_third = 0

    diff_first_second = first_val - second_val if first_val > second_val else second_val - first_val
    diff_second_third = second_val - third_val if second_val > third_val else third_val - second_val
    diff_first_third = first_val - third_val if first_val > third_val else third_val - first_val 
    
    return diff_first_second > threshold or diff_second_third > threshold or diff_first_third > threshold 

def diff_between_two_rgb_threshold(threshold, r1, g1, b1, r2, g2, b2):
    diff_r1_r2 = 0
    diff_g1_g2 = 0
    diff_b1_b2 = 0

    diff_r1_r2 = r1 - r2 if r1 > r2 else r2 - r1
    diff_g1_g2 = g1 - g2 if g1 > g2 else g2 - g1 
    diff_b1_b2 = b1 - b2 if b1 > b2 else b2 - b1

    return diff_r1_r2 > threshold or diff_g1_g2 > threshold or diff_b1_b2 > threshold

def _draw_lines(img, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]

        #here comes the logic for extracting only "valuable" lines
        line_start_x = x1
        line_start_y = y1

        red_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x - LEFT_PIXEL_DISTANCE > 0 else line_start_x][0]
        green_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x - LEFT_PIXEL_DISTANCE > 0 else line_start_x][1]
        blue_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x - LEFT_PIXEL_DISTANCE > 0 else line_start_x][2]

        red_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x + RIGHT_PIXEL_DISTANCE < WIDTH else line_start_x][0]
        green_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x + RIGHT_PIXEL_DISTANCE < WIDTH else line_start_x][1]
        blue_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x + RIGHT_PIXEL_DISTANCE < WIDTH else line_start_x][2]

        print(f"Rleft: {red_left}, gleft {green_left}, bleft: {blue_left}. Rright: {red_right}, gright {green_right}, bright: {blue_right}.")

        if not (val_diff_greater_than_threshold(GRAY_DIFFERENCE_THRESHOLD, red_left, green_left, blue_left) \
            or val_diff_greater_than_threshold(GRAY_DIFFERENCE_THRESHOLD, red_right, green_right, blue_right) \
            or diff_between_two_rgb_threshold(LEFT_RIGHT_DIFF_TRESHOLD, red_left, green_left, blue_left, red_right, green_right, blue_right)):            
            #print (f"Red value for near pixels is: {red_left}, green is {green_left}: , blue is {blue_left}")   
            #print (f"Line start x: {line_start_x}, line start y: {line_start_y}. Line end x: {line_end_x}, line end y: {line_end_y}") 

            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

def _draw_lines2(img, lines):
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        # line_start_x = x1
        # line_start_y = y1

        # red_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x > LEFT_PIXEL_DISTANCE else line_start_x][0]
        # green_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x > LEFT_PIXEL_DISTANCE else line_start_x][1]
        # blue_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x > LEFT_PIXEL_DISTANCE else line_start_x][2]

        # red_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x > RIGHT_PIXEL_DISTANCE else line_start_x][0]
        # green_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x > RIGHT_PIXEL_DISTANCE else line_start_x][1]
        # blue_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x > RIGHT_PIXEL_DISTANCE else line_start_x][2]

        # if not (value_difference_greater_than_treshold(GRAY_DIFFERENCE_TRESHOLD, red_left, green_left) \
        #     or value_difference_greater_than_treshold(GRAY_DIFFERENCE_TRESHOLD, red_left, blue_left) \
        #     or value_difference_greater_than_treshold(GRAY_DIFFERENCE_TRESHOLD, blue_left, green_left) \
        #     or value_difference_greater_than_treshold(GRAY_DIFFERENCE_TRESHOLD, red_right, green_right) \
        #     or value_difference_greater_than_treshold(GRAY_DIFFERENCE_TRESHOLD, red_right, blue_right) \
        #     or value_difference_greater_than_treshold(GRAY_DIFFERENCE_TRESHOLD, blue_right, green_right)):
        print(x1, y1, x2, y2)

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255), 2)

def calculate_distance():
    img = get_image()
    img.setflags(write=1)
    #brightness(img, 50)
    lines = detect_lines(img)

    if lines is  None:
        brightness(img, 25)
        lines = detect_lines(img)
    
    if lines is not None:
        nearest_line_distance = np.inf
        for line in lines:
            nearest_line_distance = min(nearest_line_distance,\
                                        distance_from_the_line(np.array([line[0][0], line[0][1]]), np.array([line[0][2], line[0][3]]), CENTER))
        #print(nearest_line_distance/MAX_DISTANCE_ALLOWED)
        return nearest_line_distance
    else:
        return -1

while True:
    #print(client.simGetGroundTruthKinematics().position)
    img = np.array(get_image())
    #img_copy = img.copy()
    #img.setflags(write=1)
    #brightness(img, 50)
    lines = detect_lines(img)

    if lines is not None:
        #_draw_lines2(img, lines)
        _draw_lines(img,lines)
        #for i in range (0,255, LEFT_PIXEL_DISTANCE):
        #    cv2.line(img, (i, 65), (i, 0), (0, 255, 0), 5)
    else: 
        brightness(img, 25)
        lines = detect_lines(img)
        if lines is not None:
            #_draw_lines(img, lines)
            _draw_lines(img, lines)
            #for i in range (0,255, LEFT_PIXEL_DISTANCE):
            #    cv2.line(img, (i, 65), (i, 0), (0, 255, 0), 5)

    # if lines is not None:
    #     nearest_line_distance = MAX_DISTANCE_ALLOWED
    #     for line in lines:
    #        line_start_x = line[0][0]
    #        line_start_y = line[0][1]

    #        red_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x > LEFT_PIXEL_DISTANCE else line_start_x][0]
    #        green_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x > LEFT_PIXEL_DISTANCE else line_start_x][1]
    #        blue_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x > LEFT_PIXEL_DISTANCE else line_start_x][2]

    #        red_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x > RIGHT_PIXEL_DISTANCE else line_start_x][0]
    #        green_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x > RIGHT_PIXEL_DISTANCE else line_start_x][1]
    #        blue_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x > RIGHT_PIXEL_DISTANCE else line_start_x][2]

    #        if not (val_diff_greater_than_treshold(GRAY_DIFFERENCE_TRESHOLD, red_left, green_left, blue_left) or val_diff_greater_than_treshold(GRAY_DIFFERENCE_TRESHOLD, red_right, green_right, blue_right)):
    #            nearest_line_distance = min(nearest_line_distance,\
    #                                     distance_from_the_line(np.array([line[0][0], line[0][1]]), np.array([line[0][2], line[0][3]]), CENTER))

    #     print(nearest_line_distance/MAX_DISTANCE_ALLOWED)
    # else:
    #    print("Dead")

    # if lines is not None:
    #     nearest_line_distance = np.inf
    #     # for rho, theta in lines[0]:
    #     #     a = np.cos(theta)
    #     #     b = np.sin(theta)
    #     #     x0 = a*rho
    #     #     y0 = b*rho
    #     #     x1 = int(x0 + 1000*(-b))
    #     #     y1 = int(y0 + 1000*(a))
    #     #     x2 = int(x0 - 1000*(-b))
    #     #     y2 = int(y0 - 1000*(a))
    #     #     nearest_line_distance = min(nearest_line_distance,\
    #     #                             distance_from_the_line(np.array([x1, y1]), np.array([x2, y2]), CENTER))
    #     print(nearest_line_distance/MAX_DISTANCE_ALLOWED)
    # else:
    #     print("Dead")

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cv2.imwrite(f"./{DESTINATION_FOLDER}/img{i}.png", mask)  
cv2.destroyAllWindows()