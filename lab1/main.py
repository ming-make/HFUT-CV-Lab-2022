from cv2 import cv2
import numpy as np
import math
import time

# 聚合阈值
def get_threshold(hough_space, width, height, x, y, space):
    threshold = hough_space[x][y]
    if x - space > 0 and x + space < width and y - space > 0 and y + space < height:
        tmp = []
        for i in range(x-space, x + space):
            tmp.append(max(hough_space[i][y-space:y+space]))
        threshold = max(tmp)
    return threshold

# 兴趣区域
def ROI(img,vertices):
    # 定义一个和输入图像同样大小的全黑图像mask，这个mask也称掩膜
    mask=np.zeros_like(img)
    if len(img.shape)>2:
        channel_count=img.shape[2]   # i.e. 3 or 4 根据图像而定
        ignore_mask_color=(255,)*channel_count
    else:
        ignore_mask_color=255
    # [vertices]中的点组成了多边形，将在多边形内的mask像素点保留，
    cv2.fillPoly(mask,[vertices],ignore_mask_color)
    # 与mask做"与"操作，即仅留下多边形部分的图像
    masked_image=cv2.bitwise_and(img,mask)

    return masked_image

# 直线检测
def line_detecting(path, threshold, rMin, rMax):
    # Canny边缘检测
    # Canny滞后性阈值
    cannyMin = 45
    cannyMax = 150
    image = cv2.imread(path)
    # 将原图转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 高斯滤波器
    gauss = cv2.GaussianBlur(gray, (3, 3), 0)
    # Canny边缘检测
    edges = cv2.Canny(gauss, cannyMin, cannyMax)
    cv2.imwrite('./results/line_edges_'+str(time.time())+'_.bmp', edges)
    cv2.imshow('img-canny', edges)


    # 选择兴趣区域
    # 图像像素的行列数
    [rows, cols] = edges.shape
    # 划定掩膜覆盖的区域
    left_bottom = [0, rows]
    right_bottom = [cols, rows]
    apex = [cols / 2, 170]
    vertices = np.array([left_bottom, right_bottom, apex], np.int32)
    roi_image = ROI(edges, vertices)
    cv2.imwrite('./results/line_roi_' + str(time.time()) + '_.bmp', edges)
    cv2.imshow('img-roi', roi_image)

    # Hough空间初始化
    height = roi_image.shape[0]
    width = roi_image.shape[1]
    rMax = int(math.hypot(height, width)) # 计算最大r值
    thetaMax = 360 # 将圆周分为360份，便于离散计算
    points = [[[] for k in range(thetaMax)]for theta in range(rMax)]
    hough_space = [[0 for k in range(thetaMax)] for theta in range(rMax)]
    # 投票
    for x in range(width):
        for y in range(height):
            if roi_image[y][x] == 0:
                # 无边界区域不计算
                continue
            for theta in range(0, thetaMax-1):
                # 计算r值，r =x × cos θ + y × sin θ
                r = int(x*math.cos(theta)+y*math.sin(theta))
                if rMin < r < rMax:
                    # 直线经过的每一点+1，计算该点有多少不同的直线经过
                    hough_space[r][theta] += 1
                    points[r][theta].append((x, y))
    # 找到符合大于阈值要求的直线
    # 对极坐标空间进行遍历
    for r in range(rMax):
        for theta in range(thetaMax):
            # 判断是否达到阈值要求
            if hough_space[r][theta] >= threshold:
                # 判断是否是最边缘的点
                # 为了更好的绘制效果，需要确定直线的范围，不需要中间的点
                if hough_space[r][theta] == get_threshold(hough_space, thetaMax, rMax, r, theta, 20):
                    # 计算直线参数
                    k = (-math.cos(theta)/math.sin(theta))
                    b = (r/math.sin(theta))
                    points[r][theta].sort()
                    x0 = points[r][theta][0][0]
                    x1 = points[r][theta][-1][0]
                    y0 = int(x0*k+b)
                    y1 = int(x1*k+b)
                    # 绘制直线
                    cv2.line(image, (x0, y0), (x1, y1),
                             (255, 0, 0), thickness=2)
    cv2.imshow("line-detection", image)
    cv2.imwrite("results/line_detection_" + str(time.time()) + "_.bmp", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    line_img = './assets/test9.jpg'
    line_detecting(line_img, 180, 30, 500)