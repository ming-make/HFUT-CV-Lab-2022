
import numpy as np
from scipy import ndimage
import math
import cv2

# 处理cv2窗口标题中文乱码问题
def zh_ch(string):
    return string.encode("gbk").decode(errors="ignore")

# Mode = 1 代表根据H设置阈值
# Mode = 2 代表根据Hs和Hr设置阈值
Mode = 2
imgPath = './assets/10.png'
H = 30  # 已经废弃
Hr = 30  # 聚类圈的bandwidth
Hs = 80  # RGB值的差异阈值
Iter = 5  # MeanShift向量差异的阈值
nearH = Hs * 1  # 结合操作的位置阈值
seedmap = []

# 高斯滤波器
def GaussianFilter(img):
    h, w, c = img.shape
    # 高斯滤波
    K_size = 5
    sigma = 1

    # 零填充
    pad = K_size // 2
    out = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=float)
    out[pad:pad + h, pad:pad + w] = img.copy().astype(float)

    # 定义滤波核
    K = np.zeros((K_size, K_size), dtype=float)

    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (sigma * np.sqrt(2 * np.pi))
    K /= K.sum()

    # 卷积的过程
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad + y, pad + x, ci] = np.sum(K * tmp[y:y + K_size, x:x + K_size, ci])

    out = out[pad:pad + h, pad:pad + w].astype(np.uint8)

    return out

# 等比例缩小图像
def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    if(height * width > 300 * 300):
        size_sqr = (height * width) / (300 * 300)
        size = math.sqrt(size_sqr)
        height_new = height // size
        width_new = width // size
        image_new = cv2.resize(image,(int(width_new),int(height_new)))
        return image_new

# 全局定义
#img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
img = img_resize(img)
img_back = img
img = GaussianFilter(img)
opImg = np.zeros(img.shape, np.uint8)
boundaryImg = np.zeros(img.shape, np.uint8)

# 获取所有的邻居点（颜色、位置差异度都处于阈值范围内的点）
def getNeighbors(seed, matrix,K, mode=2):
    neighbors = []
    nAppend = neighbors.append
    sqrt = math.sqrt
    for i in range(0, len(matrix)):
        if(K[i] == 0):
            continue
        cPixel = matrix[i]
        if (mode == 1):
            d = sqrt(sum((cPixel - seed) ** 2))
            if (d < H):
                nAppend(i)
        else:
            r = sqrt(sum((cPixel[:3] - seed[:3]) ** 2))
            s = sqrt(sum((cPixel[3:5] - seed[3:5]) ** 2))
            if (s < Hs and r < Hr):
                nAppend(i)
    return neighbors

# 将新获得的种子结点以及它的所有邻点记录并存储于列表seedmap中
def getnewseed(mean,neighbors, K , length):
    global seedmap
    cluster = 0
    length = length - len(neighbors)
    seed = [mean,neighbors,cluster]
    seedmap.append(seed)
    for i in neighbors:
        K[i] = 0
    return K,length

# 遍历所有的种子点，进行同类合并
def mergeseed():
    global seedmap
    sqrt = math.sqrt
    cluster = 0
    for i in range(1,len(seedmap)):
        flag = 0
        for j in range(i):
            current_seed = seedmap[i][0]
            compare_seed = seedmap[j][0]
            r = sqrt(sum((current_seed[:3] - compare_seed[:3]) ** 2))
            s = sqrt(sum((current_seed[3:5] - compare_seed[3:5]) ** 2))
            if(r < Hr and s < Hs + nearH):
                if(flag == 0):
                    seedmap[i][0][:3] = seedmap[j][0][:3]
                    seedmap[i][2] = seedmap[j][2]
                    flag = 1
                elif (flag == 1):
                    seedmap[j][0][:3] = seedmap[i][0][:3]
                    seedmap[j][2] = seedmap[i][2]

        if flag == 0:
            cluster = cluster + 1
            seedmap[i][2] = cluster


# 一张图将同类的点都涂成该类种子点的RGB值，另一张输出的图用高对比度清楚的展示分割结果
def markPixels(neighbors, mean, matrix, cluster):
    for i in neighbors:
        cPixel = matrix[i]
        x = cPixel[3]
        y = cPixel[4]
        opImg[x][y] = np.array(mean[:3], np.uint8)
        if cluster % 3 == 0:
            boundaryImg[x][y] = np.array([(cluster*10)%255,(cluster*5)%255,(cluster*15)%255],np.uint8)
        elif cluster % 3 == 1:
            boundaryImg[x][y] = np.array([(cluster * 5) % 255, (cluster * 10) % 255, (cluster * 15) % 255], np.uint8)
        else:
            boundaryImg[x][y] = np.array([(cluster * 15) % 255, (cluster * 10) % 255, (cluster * 5) % 255], np.uint8)



# 计算种子结点以及其所有邻结点的均值，mean
def calculateMean(neighbors, matrix):
    neighbors = matrix[neighbors]
    r = neighbors[:, :1]
    g = neighbors[:, 1:2]
    b = neighbors[:, 2:3]
    x = neighbors[:, 3:4]
    y = neighbors[:, 4:5]
    mean = np.array([np.mean(r), np.mean(g), np.mean(b), np.mean(x), np.mean(y)])
    return mean

# 用于获取新的种子结点
def getIndex(K):
    for i in range(len(K)):
        if(K[i]!=0):
            return i


# 将图像中所有点都置入F列表中，便于遍历
def createFeatureMatrix(img):
    h, w, d = img.shape
    F = []
    K = []
    FAppend = F.append
    for row in range(0, h):
        for col in range(0, w):
            r, g, b = img[row][col]
        #    vote = {}
            FAppend([r, g, b, row, col])
            K.append(1)
    F = np.array(F)
    length = len(F)
    K = np.array(K)
    return F , K ,length



# meanshift总步骤
def performMeanShift(img):
    clusters = 0
    F , K ,length= createFeatureMatrix(img)
    tempx = 0
    tempy = 0
    while (length > 0):
        print('还剩下的未分类像素点 : ' + str(length))
        print(1)
        randomIndex = getIndex(K)
        seed = F[randomIndex]

        count = 10
        while(count):
            initialMean = seed
            print('该中心meanshift次数:'+str(count))
            neighbors = getNeighbors(seed, F,K, Mode)
            print('找到相似点数目 :: ' + str(len(neighbors)))

            if (len(neighbors) == 1):
                K,length =getnewseed(seed,neighbors,K,length)

                break

            mean = calculateMean(neighbors, F)
            meanShift = abs(mean - initialMean)
            if (np.mean(meanShift) < Iter):  #若mean和seed差异小于阈值，则跳出循环
                K,length = getnewseed(mean,neighbors,K,length)
                break
            else :  #若大于阈值，为seed赋予mean的值（meanshift过程），继续重新找邻结点
                count = count - 1
                seed = mean
    mergeseed()    #可归为一类的不同区域归为一类
    for seed in seedmap:
        markPixels(seed[1],seed[0],F,seed[2])  #绘制图像
        if(clusters < seed[2]):
            clusters = seed[2]

    return clusters


# Method main
def main():
    clusters = performMeanShift(img)
    origlabelledImage, orignumobjects = ndimage.label(opImg)

    cv2.imshow(zh_ch('original_img'), img_back)
    cv2.imshow(zh_ch('gaussian_img'), img)
    cv2.imshow(zh_ch('clustering_img'), opImg)
    cv2.imshow(zh_ch('segmentation_img'), boundaryImg)

    name = imgPath[9]
    if(imgPath[10]!='.'):
        name += imgPath[10]
    cv2.imwrite("results/segmentation_img_" + name + "_.bmp", boundaryImg)

    print('共找到聚类个数 : ', clusters)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()