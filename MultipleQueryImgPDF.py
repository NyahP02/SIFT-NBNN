##This file takes the query img and computes a PDF ##
import os
import csv
import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image as im 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math
import glob


shapelabel = ['shape']
imagelabel =['Image']
columns = ['entry' + str(i) for i in range (128)]
descColumns = imagelabel + shapelabel + columns
queryImgMemory = pd.DataFrame(columns=descColumns)


fileName = 'QueryImgDescriptors.csv'

image_directory = (r'C:\Users\nyahp\Downloads\ay24_nnbn-main\ay24_nnbn-main\OriginalImageFiles')
query_images = glob.glob(os.path.join(image_directory, '*.png'))

for image in query_images:
    #pulls associated image name 
    image_name = os.path.basename(image)

    queryimg = cv.imread(image, cv.IMREAD_UNCHANGED)
    queryarray = np.array(queryimg) #change into array for OpenCV
    #print(queryarray)

    # #queryimg2 = im.fromarray(queryarray.astype(np.uint8) * 255)  # Multiply by 255 to scale 0-1 to 0-255

    newsize = (1000, 1000) 
    queryimgResized= cv.resize(queryimg,newsize,interpolation=cv.INTER_NEAREST)
    # queryimg.save(f'QueryImg.png')
    queryimg_np = np.array(queryimgResized)

    if queryimg_np.ndim == 2:  # If the image is grayscale
        queryimg_np = cv.cvtColor(queryimg_np, cv.COLOR_GRAY2BGR)
        
    width, height = queryimgResized.shape[:2]
    #print(width,height)
    gray= cv.cvtColor(queryimg_np,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    _ , desc = sift.compute(gray,kp)

    # print('Descriptors for Query img')
    # print(desc)
    # print("\n") 
    # #Returns an image of the queryImg w/ descriptors
    # queryimg_with_kp=cv.drawKeypoints(gray,kp,queryimg_np,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # cv.imwrite('sift_keypoints.jpg',queryimg_with_kp)
    # cv.imshow('Query image', queryimg_with_kp)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    if desc is not None: 
            all_desc = []
            for d in desc:
                try:
                    all_desc.append([image_name, 'query'] + d.tolist())
                    #print(all_desc)
                except Exception:
                        pass
        
            df_Temp = pd.DataFrame(all_desc, columns=descColumns)
            queryImgMemory = pd.concat([queryImgMemory, df_Temp],axis=0, ignore_index= True)
    #print(queryImgMemory)


trainingData = pd.read_csv(r'C:\Users\nyahp\Downloads\ay24_nnbn-main\ay24_nnbn-main\squareTriangleDescriptors.csv')
#print(trainingData)


#First Row of 
trainingRow = trainingData.iloc[1, 2:]
queryImgRow = queryImgMemory.iloc[1,2:]
#Add .tolist to see in array format
#print(triangleRow)
#print(queryImgRow)

oneRowSub= queryImgRow - trainingRow
scaledOneRowSub = .0001*oneRowSub
#print(scaledOneRowSub)
#print(oneRowSub)


########### PDF!! ###########


mean128 = np.zeros(128) # Mean vector for a 2D distribution
cov128 = np.diag(np.ones(128)*0.1)  # Covariance matrix (identity matrix for independent variables)
#diagonal matrix 
rv128 = multivariate_normal(mean128, cov128) #K
#print(rv128.pdf(np.zeros(128)))

list_est_prob_i= []
for i in range(len(queryImgMemory)):
    
    queryImgRow = queryImgMemory.iloc[i, 2:]
    
    # stores the results
    prob_query_i_descc = []
    
    # Inner loop: Subtract each value in queryImgRow from the corresponding value in the triangleRow
    trianglememory= trainingData[trainingData['shape']=='triangle']
    for j in range(len(trianglememory)):
        trainingRow = trianglememory.iloc[j, 2:]
        diff_scaled = (queryImgRow - trainingRow)*1e-3
        #print(diff_scaled)
        density = rv128.pdf(diff_scaled)
        #print(density)
        prob_query_i_descc.append(density)
        
    est_prob_i = sum(prob_query_i_descc)/len(trianglememory)
    list_est_prob_i.append(math.log(est_prob_i))
print(sum(list_est_prob_i))
exit()
# sqr_list_est_prob_i= []
# for i in range(len(queryImgMemory)):
    
#     queryImgRow = queryImgMemory.iloc[i, 1:]
    
#     # stores the results
#     prob_query_i_descc = []
    
#     # Inner loop: Subtract each value in queryImgRow from the corresponding value in the triangleRow
#     for j in range(len(squareMemory)):
#         squareRow = squareMemory.iloc[j, 1:]
#         diff_scaled = (queryImgRow - squareRow)*1e-3
#         #print(diff_scaled)
#         density = rv128.pdf(diff_scaled)
#         #print(density)
#         prob_query_i_descc.append(density)
        
#     est_prob_i = sum(prob_query_i_descc)/len(squareMemory)
#     sqr_list_est_prob_i.append(math.log(est_prob_i))
# print(sum(sqr_list_est_prob_i))

exit()



mean128 = np.zeros(128) # Mean vector for a 2D distribution
cov128 = np.diag(np.ones(128))  # Covariance matrix (identity matrix for independent variables)
#diagonal matrix 
rv128 = multivariate_normal(mean128, cov128)
x128= np.ones(128) * 1.0e-30
pdf_value128 = rv128.pdf(x128)
#print(pdf_value128)

mean3 = [0, 0,0]  # Mean vector for a 2D distribution
cov3 = [[1, 0, 0], [0, 1, 0], [0,0,1]] 
rv3 = multivariate_normal(mean3, cov3) 
x3=np.ones(3)* 5.0e-30
pdf_value3 =rv3.pdf(x3)
print(pdf_value3)
exit()

mean4 = [0, 0,0,0]  # Mean vector for a 2D distribution
cov4 = [[1, 0, 0,0], [0, 1, 0,0], [0,0,1,0], [0,0,0,1]] 
rv4 = multivariate_normal(mean4, cov4) 
x4=np.ones(4)*.1
pdf_value4 =rv4.pdf(x4)
print(pdf_value4)
exit()
# 
#
# 
# #Creates CSV File of Img Query Descriptors
# if os.path.isfile(fileName):
#     queryImgMemory.to_csv(fileName, index=False, header=False, mode='a')
# else:
#     queryImgMemory.to_csv(fileName, index=False, header=True, mode='a')