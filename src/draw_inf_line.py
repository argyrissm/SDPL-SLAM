import numpy as np
import cv2

def plot_infinite_lines(mv_stat_infinite_lines):
    # Create a white image of 640x480 pixels
    img_cpy = np.ones((480, 640), dtype=np.uint8) * 255

    line = mv_stat_infinite_lines

    # Convert numpy array to cv::Vec3f
    converted_line = (line[0], line[1], line[2])

    a = line[0]
    b = line[1]
    c = line[2]

    p1 = (0, -c/b)
    p2 = (img_cpy.shape[1], -(c + a*img_cpy.shape[1])/b)

    cv2.line(img_cpy, p1, p2, (0,0,255), 1, 8)


    cv2.imshow("infinite lines", img_cpy)
    cv2.waitKey(0)


plot_infinite_lines([-0.00053078661934172916, -0.0036797742991804454, 0.99999308873945303])