import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Returns the DOG pyramid for 4 octaves
# Successive LOG scales differ by factor sqrt(2)
def get_dog_pyramid(img,scale):
    blur = cv2.GaussianBlur(img,(0,0),scale).astype(int)
    dog = []
    scale_space = [blur]
    for i in range(4):
        scale = scale * math.sqrt(2)
        blur = cv2.GaussianBlur(img,(0,0),scale).astype(int)
        scale_space.append(blur)

        dog.append(scale_space[-1]-scale_space[-2])
    cv2.waitKey()
    cv2.destroyAllWindows()
    return [scale_space,dog]

# Checks whether a point keypoint or not by looking at surrounding 26 points
def isKeypoint(bottom,middle,top,i,j):
    maxi_middle = 0
    mini_middle = 255
    for k in range(i-1,i+2,1):
        for l in range(j-1,j+2,1):
            if(i==k and j==l):
                continue
            maxi_middle = max(maxi_middle,middle[k][l])
            mini_middle = min(mini_middle,middle[k][l])

    maxi = np.max(bottom[i-1:i+2,j-1:j+2])
    maxi = max(maxi,np.max(top[i-1:i+2,j-1:j+2]))
    maxi = max(maxi,maxi_middle)
    
    mini = np.max(bottom[i-1:i+2,j-1:j+2])
    mini = min(mini,np.max(top[i-1:i+2,j-1:j+2]))
    mini = min(mini_middle,mini)

    
    if(middle[i,j] < mini or middle[i,j] > maxi):
        return True
    return False

# Returns SIFT keypoints of img
def SIFT(img):
    img = img.astype('uint8')
    original = img
    downsampled = img

    octaves = {}
    start_scale = 0.707107
    num_octaves = 4

    scale = start_scale


    for i in range(num_octaves):
        (h,w) = downsampled.shape
        print((h,w),scale)
        octaves[scale] = get_dog_pyramid(downsampled,scale)
        
        scale = scale*(math.sqrt(2)**2)
        downsampled = cv2.resize(downsampled, (h//2,w//2), interpolation=cv2.INTER_AREA)

    # Using only the first octave for detecting the feature points
    keypoints = []
    dog = octaves[start_scale][1]
    (h,w) = img.shape
    for bottom,middle,top in zip(dog,dog[1:],dog[2:]):
        for i in range(1,h-1,1):
            for j in range(1,w-1,1):
                if abs(middle[i,j]) >thresh and isKeypoint(bottom,middle,top,i,j) == True:
                    keypoints.append((j,i))

    # Plot the keypoints on image
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    for point in keypoints:
        img = cv2.circle(img, point, radius=2, color=(0, 0, 255), thickness=-1)
    return keypoints,img

# Use thresh = 5 for books.png
# Use thresh = 10 fro building.png
img = cv2.imread("data/building.png",0)
#img = cv2.imread("data/books.png",0)
thresh = 10
keypoints1,img1 = SIFT(img)
plt.imshow(img1)
#plt.savefig('figures/books/original.png')
keypoints2,img2 = SIFT(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
keypoints3,img3 = SIFT(cv2.rotate(img, cv2.ROTATE_180))
keypoints4,img4 = SIFT(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))


f, ax = plt.subplots(2,2,figsize = (10,10))
f.suptitle("Rotation")
ax[0,0].set_title("Original -- Keypoints = "+str(len(keypoints1)))
ax[0,1].set_title("ROTATE_90_CLOCKWISE -- Keypoints = "+str(len(keypoints2)))
ax[1,0].set_title("ROTATE_180 -- Keypoints = "+str(len(keypoints3)))
ax[1,1].set_title("ROTATE_90_COUNTERCLOCKWISE -- Keypoints = "+str(len(keypoints4)))


ax[0,0].imshow(img1)
ax[0,1].imshow(img2)
ax[1,0].imshow(img3)
ax[1,1].imshow(img4)
#plt.savefig('figures/rotation.png')
plt.show()


keypoints2,img2 = SIFT(cv2.resize(img, (img.shape[0]//2,img.shape[1]//2)))
keypoints3,img3 = SIFT(cv2.resize(img, (img.shape[0]*2,img.shape[1]*2)))
#keypoints4,img4 = SIFT(cv2.resize(img, (1200,1200)))

plt.figure(figsize = (12,8))
plt.title(" Upscaling -- Keypoints = "+str(len(keypoints3)))
plt.imshow(img3)
#plt.savefig('figures/Upscaling.png')
plt.show()

plt.title(" Downscaling -- Keypoints = "+str(len(keypoints2)))
plt.imshow(img2)
#plt.savefig('figures/Downscaling.png')
plt.show()

# Increasing the scale of blur decreases the #keypoints
keypoints2,img2 = SIFT(cv2.GaussianBlur(img,(3,3),2))
keypoints3,img3 = SIFT(cv2.GaussianBlur(img,(3,3),4))
keypoints4,img4 = SIFT(cv2.GaussianBlur(img,(3,3),7))


f, ax = plt.subplots(2,2,figsize = (10,10))
f.suptitle("Guassian_Blur")
ax[0,0].set_title("Original -- Keypoints = "+str(len(keypoints1)))
ax[0,1].set_title("Sigma = 2 -- Keypoints = "+str(len(keypoints2)))
ax[1,0].set_title("Sigma = 4 -- Keypoints = "+str(len(keypoints3)))
ax[1,1].set_title("Sigma = 7 -- Keypoints = "+str(len(keypoints4)))


ax[0,0].imshow(img1)
ax[0,1].imshow(img2)
ax[1,0].imshow(img3)
ax[1,1].imshow(img4)
#plt.savefig('figures/Guassian_blur.png')
plt.show()


# Increasing scale of guassian noise increases the False Positive Keypoints
guass_noise = np.random.normal(scale = 1,size = img.shape)
keypoints2,img2 = SIFT(img+guass_noise)

guass_noise = np.random.normal(scale = 3,size = img.shape)
keypoints3,img3 = SIFT(img+guass_noise)

guass_noise = np.random.normal(scale = 15,size = img.shape)
keypoints4,img4 = SIFT(img+guass_noise)


f, ax = plt.subplots(2,2,figsize = (10,10))
f.suptitle("Guassian Noise")
ax[0,0].set_title("Original -- Keypoints = "+str(len(keypoints1)))
ax[0,1].set_title("Sigma = 5 -- Keypoints = "+str(len(keypoints2)))
ax[1,0].set_title("Sigma = 10 -- Keypoints = "+str(len(keypoints3)))
ax[1,1].set_title("Sigma = 15 -- Keypoints = "+str(len(keypoints4)))


ax[0,0].imshow(img1)
ax[0,1].imshow(img2)
ax[1,0].imshow(img3)
ax[1,1].imshow(img4)
#plt.savefig('figures/Guassian_Noise.png')
plt.show()

