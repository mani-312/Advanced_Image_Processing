import bm3d
from skimage import io, img_as_float,metrics
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
img = cv2.imread('../lighthouse2.bmp',cv2.IMREAD_GRAYSCALE)

# Add Guassian noise
var = 100
noisy_img = img + np.random.normal(loc=0, scale=np.sqrt(var), size=img.shape)

# Convert the image to a float representation
#noisy_img = img_as_float(noisy_img)


# Apply the first stage of the BM3D algorithm
psd = 10
first_stage = bm3d.bm3d(noisy_img, sigma_psd=psd, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

# Apply the second stage of the BM3D algorithm
second_stage = bm3d.bm3d(noisy_img, sigma_psd=psd, stage_arg=first_stage)

# Apply whole pipeline
final = bm3d.bm3d(noisy_img, sigma_psd=psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)

# Here final and second_stage are same

mse_first_stage = metrics.mean_squared_error(img, first_stage)
mse_second_stage = metrics.mean_squared_error(img, second_stage)
mse_final = metrics.mean_squared_error(img, final)

print('MSE between original image and first stage: {:.10f}'.format(mse_first_stage))
print('MSE between original image and second stage: {:.10f}'.format(mse_second_stage))
print('MSE between original image and second stage: {:.10f}'.format(mse_final))

cv2.imshow("Original",noisy_img.astype('uint8'))
cv2.imshow("First_stage",first_stage.astype('uint8'))
cv2.imshow("Second_stage",second_stage.astype('uint8'))
cv2.imshow("Final",final.astype('uint8'))

cv2.imwrite("Original.bmp",noisy_img.astype('uint8'))
cv2.imwrite("First_stage.bmp",first_stage.astype('uint8'))
cv2.imwrite("Second_stage.bmp",second_stage.astype('uint8'))
#cv2.imwrite("Final",final.astype('uint8'))


sigma = np.arange(1,20,1)
first_stage_MSE = []
second_stage_MSE = []
for sg in sigma:
    first_stage = bm3d.bm3d(noisy_img, sigma_psd=sg, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    second_stage = bm3d.bm3d(noisy_img, sigma_psd=sg, stage_arg=first_stage)
    
    first_stage_MSE.append(metrics.mean_squared_error(img, first_stage))
    second_stage_MSE.append(metrics.mean_squared_error(img, second_stage))
    print(sg,'done')

plt.plot(sigma*sigma,first_stage_MSE,'-o')
plt.plot(sigma*sigma,second_stage_MSE,'-o')
plt.xlabel('Sigma_square')
plt.ylabel('MSE')
plt.legend(['First_stage','Second_stage'])
plt.title('Sigma_Square vs MSE')

plt.savefig('SigmavsMSE.png')
plt.show()


