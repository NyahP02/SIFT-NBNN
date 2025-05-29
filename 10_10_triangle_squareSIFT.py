# This file creates a dataset in a csv file in the /data folder that defines many squares and triangles
#All shapes are outlined and not filled

import os
import pandas as pd
import numpy as np
import cv2 as cv
from PIL import Image as im 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


_numRowCol = 10 #Increases size of grid
shapelabel = ['shape']
imagelabel = ['Image']
columns = ['d' + str(i) for i in range (128)]
descColumns = imagelabel + shapelabel + columns
squareMemory = pd.DataFrame(columns=descColumns)
triangleMemory = pd.DataFrame(columns=descColumns)
queryMemory = pd.DataFrame(columns=descColumns)
shapeData = pd.concat([squareMemory, triangleMemory], axis=1)
mainMemory = pd.DataFrame()


#A 1d array representing the 10x10 grid for entering shapes; created with all 0
grid0 = [0] * _numRowCol**2

#newimg = np.repeat(img, 50) 

fileName = 'squareTriangleDescriptors.csv'
fileName2 = 'triangleDescriptors.csv'
fileName3 = 'squareDescriptors.csv'

#If the file exists, delete it to generate the data over again
if os.path.isfile(fileName):
    print('Successful')
    os.remove(fileName)
else:
    print('failed')

#Beginning of Creation of squares
#Creates an array that will store all of the different sized squares starting at the top left of the input matrix
_startingSquares = []
#Creates different sized squares
for i in range(2, _numRowCol + 1):
    square = list(grid0)
    for j in range(i):
        square[j] = 1
        square[j + 10 * i - 10] = 1
        square[j * 10] = 1
        square[j * 10 + i - 1] = 1
    _startingSquares.append(square)
# squareShape= [np.array(_startingSquares[x:x+10]).reshape(-1, 10)for x in range(0, len(_startingSquares), 10)]



#Prints array of squares, converts each row to numpy array, resizes it and applies SIFT Keypoint to image



#Creates an array that will store all of the different sized triangles starting at the top left of the input matrix
_startingTriangles = []

#Base 3 triangels

triangle = list(grid0)
triangle[1] = 1

_startingTriangles.append(triangle)


triangle = list(grid0)
triangle[0] = 1
triangle[10] = 1
triangle[11] = 1
triangle[20] = 1
_startingTriangles.append(triangle)

triangle = list(grid0)
triangle[0] = 1
triangle[1] = 1
triangle[2] = 1
triangle[11] = 1
_startingTriangles.append(triangle)

triangle = list(grid0)
triangle[1] = 1
triangle[10] = 1
triangle[11] = 1
triangle[21] = 1
_startingTriangles.append(triangle)

#Base 5 triangels

triangle = list(grid0)
triangle[2] = 1
triangle[11] = 1
triangle[13] = 1
triangle[20] = 1
triangle[21] = 1
triangle[22] = 1
triangle[23] = 1
triangle[24] = 1
_startingTriangles.append(triangle)

triangle = list(grid0)
triangle[0] = 1
triangle[10] = 1
triangle[11] = 1
triangle[20] = 1
triangle[22] = 1
triangle[30] = 1
triangle[31] = 1
triangle[40] = 1
_startingTriangles.append(triangle)

triangle = list(grid0)
triangle[2] = 1
triangle[11] = 1
triangle[12] = 1
triangle[20] = 1
triangle[22] = 1
triangle[31] = 1
triangle[32] = 1
triangle[42] = 1
_startingTriangles.append(triangle)

#Base 7 triangles

triangle = list(grid0)
triangle[3] = 1
triangle[12] = 1
triangle[14] = 1
triangle[21] = 1
triangle[25] = 1
triangle[30] = 1
triangle[31] = 1
triangle[32] = 1
triangle[33] = 1
triangle[34] = 1
triangle[35] = 1
triangle[36] = 1
_startingTriangles.append(triangle)

triangle = list(grid0)
triangle[3] = 1
triangle[12] = 1
triangle[13] = 1
triangle[21] = 1
triangle[23] = 1
triangle[30] = 1
triangle[33] = 1
triangle[41] = 1
triangle[43] = 1
triangle[52] = 1
triangle[53] = 1
triangle[63] = 1
_startingTriangles.append(triangle)

triangle = list(grid0)
triangle[0] = 1
triangle[1] = 1
triangle[2] = 1
triangle[3] = 1
triangle[4] = 1
triangle[5] = 1
triangle[6] = 1
triangle[11] = 1
triangle[15] = 1
triangle[22] = 1
triangle[24] = 1
triangle[33] = 1
_startingTriangles.append(triangle)

triangle = list(grid0)
triangle[0] = 1
triangle[10] = 1
triangle[11] = 1
triangle[20] = 1
triangle[22] = 1
triangle[30] = 1
triangle[33] = 1
triangle[40] = 1
triangle[42] = 1
triangle[50] = 1
triangle[51] = 1
triangle[60] = 1
_startingTriangles.append(triangle)

#Base 9 triangles

triangle = list(grid0)
triangle[4] = 1
triangle[13] = 1
triangle[15] = 1
triangle[22] = 1
triangle[26] = 1
triangle[31] = 1
triangle[37] = 1
triangle[40] = 1
triangle[41] = 1
triangle[42] = 1
triangle[43] = 1
triangle[44] = 1
triangle[45] = 1
triangle[46] = 1
triangle[47] = 1
triangle[48] = 1
_startingTriangles.append(triangle)

triangle = list(grid0)
triangle[4] = 1
triangle[13] = 1
triangle[14] = 1
triangle[22] = 1
triangle[24] = 1
triangle[31] = 1
triangle[34] = 1
triangle[40] = 1
triangle[44] = 1
triangle[51] = 1
triangle[54] = 1
triangle[62] = 1
triangle[64] = 1
triangle[73] = 1
triangle[74] = 1
triangle[84] = 1
_startingTriangles.append(triangle)

triangle = list(grid0)
triangle[0] = 1
triangle[1] = 1
triangle[2] = 1
triangle[3] = 1
triangle[4] = 1
triangle[5] = 1
triangle[6] = 1
triangle[7] = 1
triangle[8] = 1
triangle[11] = 1
triangle[17] = 1
triangle[22] = 1
triangle[26] = 1
triangle[33] = 1
triangle[35] = 1
triangle[44] = 1
_startingTriangles.append(triangle)

triangle = list(grid0)
triangle[0] = 1
triangle[10] = 1
triangle[11] = 1
triangle[20] = 1
triangle[22] = 1
triangle[30] = 1
triangle[33] = 1
triangle[40] = 1
triangle[44] = 1
triangle[50] = 1
triangle[53] = 1
triangle[60] = 1
triangle[62] = 1
triangle[70] = 1
triangle[71] = 1
triangle[80] = 1
_startingTriangles.append(triangle)





_nodeNames = list("x" + str(i) for i in range(_numRowCol**2))

#print(_dataFrame)

_dataFrame = pd.DataFrame(columns=['Image', 'shape'] + _nodeNames)





#Functions for moving shapes around the grid

def shiftRight(grid):
    enable = False
    outOfBounds = False
    active = grid.count(1)

    for i in range(len(grid)):
        if grid[i] == 1:
            if not enable:
                grid[i] = 0
            enable = True
        else:
            if enable:
                grid[i] = 1
                if i % _numRowCol == 0:
                    outOfBounds = True
            enable = False

    if active != grid.count(1):
        outOfBounds = True
    
    return grid, outOfBounds

def shiftDown(grid):
    enableList = []
    outOfBounds = False

    for i in range(len(grid)):
        if grid[i] == 1:
            if i not in enableList:
                grid[i] = 0
            enableList.append(i+10)
        else:
            if i in enableList:
                grid[i] = 1
                if i > (_numRowCol**2) - 1:
                    outOfBounds = True
                enableList.remove(i)
    
    return grid, outOfBounds


#Loops that move the shape around the grid

for square in _startingSquares:
    grid = list(square)
    _active = grid.count(1)
    for i in range(_numRowCol):
        beforeRightShift = list(grid)
        for j in range(_numRowCol):

            square = {'shape': 0}

            for j in range(_numRowCol**2):
                square[_nodeNames[j]] = grid[j]

            _dataFrame.loc[len(_dataFrame)] = square
            grid, outOfBounds = shiftRight(grid)
            
            if(outOfBounds):
                break
        grid, outOfBounds = shiftDown(beforeRightShift)

        if outOfBounds or _active != grid.count(1):
            break

for i in range(len(_dataFrame)):
    # square = _startingSquares[i]
    # print(f"Square {i + 2}:")
    # print(square)
    # print("\n")
    row = _dataFrame.iloc[i]
    
    # Extract the grid configuration from the DataFrame for the current triangle
    grid_values = [row[node] for node in _nodeNames]

    # Convert the current square to a NumPy array and save as an image
    if row['shape'] == 0:
        square_array = np.array(grid_values).reshape((10, 10))

        #This is unneccessary for CV but is necessary for .resize function in PIL (Converts array to PIL Image)
        img = im.fromarray(square_array.astype(np.uint8) * 255)  # Multiply by 255 to scale 0-1 to 0-255
        output_dir = 'squares_output'
        os.makedirs(output_dir, exist_ok=True)
        img.save(os.path.join(output_dir,f'mySquare_{i + 1}.png'))

        #Alternate way to resize in CV
        #img_np = cv.resize(square_array.astype(np.uint8) * 255, (1000, 1000), interpolation=cv.INTER_NEAREST)

        #resize the new image 
        newsize = (1000, 1000) 
        imgResized= img.resize(newsize)

        #Resized Image MUST be converted BACK to a NP array
        img_np = np.array(imgResized)

        # Convert grayscale image to BGR (3 channels) for OpenCV
        if img_np.ndim == 2:  # If the image is grayscale
            img_np = cv.cvtColor(img_np, cv.COLOR_GRAY2BGR)
        
        width, height = img.size 
        #print(width,height)
        gray= cv.cvtColor(img_np,cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp = sift.detect(gray,None)
        _ , desc = sift.compute(gray,kp)

        
        # print(f"Descriptors for Square {i + 2}:")
        # print(desc)
        # print("\n") 
        

        # img_with_kp=cv.drawKeypoints(gray,kp,img_np,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imwrite('sift_keypoints.jpg',img_with_kp)
        # cv.imshow('square', img_with_kp)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        
        
        #print(df1)
        ##### IMPORTANT ########## 
    if desc is not None: 
        all_desc = []
        for d in desc:
            try:
                all_desc.append([f'square_{i + 1}','square' ] + d.tolist())
                #print(all_desc)
            except Exception:
                    pass
    

        df_Temp = pd.DataFrame(all_desc, columns=descColumns)
        squareMemory = pd.concat([squareMemory, df_Temp],axis=0, ignore_index= True)
        # img.save(r'C:\Users\nyahp\Documents\SIFT - PDF Code\New folder')
squareRow = squareMemory.iloc[0, 1:].tolist()

for triangle in _startingTriangles:
    grid = list(triangle)
    _active = grid.count(1)
    for i in range(_numRowCol):
        beforeRightShift = list(grid)
        for j in range(_numRowCol):

            triangle = {'shape': 1}

            for j in range(_numRowCol**2):
                triangle[_nodeNames[j]] = grid[j]

            _dataFrame.loc[len(_dataFrame)] = triangle
            grid, outOfBounds = shiftRight(grid)
            
            if(outOfBounds):
                break
        grid, outOfBounds = shiftDown(beforeRightShift)

        if outOfBounds or _active != grid.count(1):
            break

for i in range(len(_dataFrame)):
    # triangle = _startingTriangles[i]
    # print(f"Triangle {i + 2}:")
    # print(triangle)
    # print("\n")
    row = _dataFrame.iloc[i]
    
    # Extract the grid configuration from the DataFrame for the current triangle
    grid_values = [row[node] for node in _nodeNames]

    if row['shape'] == 1:# Convert the current triangle to a NumPy array and save as an image
        triangle_array = np.array(grid_values).reshape((10, 10)) #.Reshape affects your pictures heavily

        #This is unneccessary for CV but is necessary for .resize function in PIL (Converts array to PIL Image)
        img2 = im.fromarray(triangle_array.astype(np.uint8) * 255)  # Multiply by 255 to scale 0-1 to 0-255
        output_dir = 'triangles_output'
        os.makedirs(output_dir, exist_ok=True)
        img2.save(os.path.join(output_dir,f'myTriangle_{i + 1}.png'))
        #Alternate way to resize in CV
        #img_np = cv.resize(square_array.astype(np.uint8) * 255, (1000, 1000), interpolation=cv.INTER_NEAREST)

        #resize the new image 
        newsize = (1000, 1000) 
        img2Resized= img2.resize(newsize)

        #Resized Image MUST be converted BACK to a NP array
        img2_np = np.array(img2Resized)

        # Convert grayscale image to BGR (3 channels) for OpenCV
        if img2_np.ndim == 2:  # If the image is grayscale
            img2_np = cv.cvtColor(img2_np, cv.COLOR_GRAY2BGR)
        
        width, height = img2.size 
        #print(width,height)
        gray= cv.cvtColor(img2_np,cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp = sift.detect(gray,None)
        _ , desc = sift.compute(gray,kp)
        #Save the descriptors to a file
        #np.savetxt(f'descriptors_{i + 1}.txt', desc)

        img_with_kp=cv.drawKeypoints(gray,kp,img2_np,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # print(f"Descriptors for Triangle {i + 2}:")
        # print(desc)
        # print("\n") #represent normalized gradient magnitudes within the patch.
        
        
        # cv.imwrite('sift_keypoints.jpg',img_with_kp)
        # cv.imshow('triangle', img_with_kp)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    if desc is not None: 
        all_desc = []
        for d in desc:
            try:
                all_desc.append([f'Triangle_{i + 1}', 'triangle'] + d.tolist())
                #print(all_desc)
            except Exception:
                pass

    df_Temp = pd.DataFrame(all_desc, columns=descColumns)
    triangleMemory = pd.concat([triangleMemory, df_Temp],axis=0, ignore_index= True)
triangleRow = triangleMemory.iloc[0, 1:].tolist() #gives the first row of the triangle dataframe 
#print(triangleRow)
#print(triangleMemory)

frames = [squareMemory, triangleMemory]
mainMemory = pd.concat(frames) #You now have Training Data in one place!!!
#print(mainMemory)

_dataFrame = pd.concat(frames)




if os.path.isfile(fileName):
    mainMemory.to_csv(fileName, index=False, header=False, mode='a')
else:
    mainMemory.to_csv(fileName, index=False, header=True, mode='a')

if os.path.isfile(fileName2):
    triangleMemory.to_csv(fileName2, index=False, header=False, mode='a')
else:
    triangleMemory.to_csv(fileName2, index=False, header=True, mode='a')


if os.path.isfile(fileName3):
    squareMemory.to_csv(fileName3, index=False, header=False, mode='a')
else:
    squareMemory.to_csv(fileName3, index=False, header=True, mode='a')