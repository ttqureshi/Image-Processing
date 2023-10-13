import cv2 as cv
import numpy as np

image = cv.imread('image.jpg')
cv.imshow("Image",image)
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Gray Image",image_gray)





cv.imwrite("gray image.jpg", image_gray)



print(image.shape)
print(image_gray.shape)
print(len(image.flatten()))
print(len(image_gray.flatten()))


# scaling the image by factor of 0.5, since we have to copy the pixel values from old image to the scaled image so we will make the transformation matrix M with the scale factor of (1/0.5 = 2)
M = np.array([
    [2,0],
    [0,2]])

# print(M)

height, width = image_gray.shape
center_x = width//2
center_y = height//2
print(f"Center at: ({center_x},{center_y})")

# scale about the point c
c = np.array([center_x,center_y]).reshape(-1,1)
print(c)

# p = np.dot(M,c.reshape(-1,1))
# print(p.shape)
# print(p)
# print(c + p)
# p = p.flatten()
# print(p,p.shape)


scaled_image = np.zeros((image_gray.shape[0]//2,image_gray.shape[1]//2))
cv.imshow("Blank",scaled_image)
print(">>>")
print(scaled_image.shape)


for i in range(len(scaled_image)):
    for j in range(len(scaled_image[i])):
        n = np.array([i,j]).reshape(-1,1)
        sub = (n - c)
        dotprod = np.dot(M,sub)
        n_hat = c + dotprod
        # n_hat = n_hat.flatten()
        # print(n[0,0],n[1,0])
        print(f"image_gray[{round(n_hat[0,0])}, {round(n_hat[1,0])}]: {image_gray[round(n_hat[0,0]), round(n_hat[1,0])]}")
        x = min(max(int(n_hat[0,0]),0),image_gray.shape[0]-1)
        y = min(max(int(n_hat[1,0]),0),image_gray.shape[1]-1)
        scaled_image[n[0,0],n[1,0]] = image_gray[x,y]

cv.imshow("Scaled Image",scaled_image)
        


cv.waitKey(0)
cv.destroyAllWindows()