import numpy as np
import skimage.io as io
from matplotlib import pyplot as plt
import cv2


def JPEG_compress(image,q_table,factor = 1, view = False, prints = True):
    m, n = image.shape

    #quantized stores the quantized values to be encoded
    quantized = np.zeros((m, n))

    #reconstructed image
    reconstructed = np.zeros((m, n))

    for i in range(0,m,8):
        for j in range(0,n,8):
            # 8*8 block formation
            image_patch=image[i:i+8,j:j+8]
            dct_image_patch=cv2.dct(np.float32(image_patch))
            quant_patch=np.floor(dct_image_patch / q_table +0.5 )

            #storing each quantized patch in quantized
            quantized[i:i+8,j:j+8]=quant_patch

            #storing each reconstructed patch in reconstructed
            reconstructed[i:i+8,j:j+8]=cv2.idct(np.float32(quant_patch * q_table))
            
    # bringing the range to 0-255 for computation of MSE
    reconstructed = (reconstructed- np.min(reconstructed)) / (np.max(reconstructed) - np.min(reconstructed)) * 255
    reconstructed = reconstructed.astype(np.uint8)

    # MSE computation
    mse = np.square(np.subtract(image, reconstructed)).mean()
        

    # output file size computaion in KBs and compression ratio calculation
    encoded_string = ""
    for i in range(m):
        for j in range(n):
            encoded_string = encoded_string + Lossles_source_encoding(quantized[i, j])

    compressed_size = len(encoded_string) / (1024*8)
    comp_ratio = (m*n*8)/len(encoded_string)

    if prints:
        print('MSE for case of given Q_table factor = ',factor,' is:   ', mse)
        print('Size of the compressed file is in Kilobyte:  ',compressed_size)
        print('Compression ratio for case 1 is   :',comp_ratio)


    if view:

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,10))

        # Plot the data on the first subplot
        axs[0].imshow(image,cmap='gray')
        axs[0].set_title('Original image')

        # Plot the data on the second subplot
        axs[1].imshow(reconstructed,cmap='gray')
        axs[1].set_title('Reconstructed, factor = '+str(factor))
        plt.savefig("Original_Reconstructed.jpg")
        # Show the plot
        plt.show()
    
    return mse, compressed_size

def Lossles_source_encoding(i):
    # each s value for defining the group of values in ecodeding table
    s = [0, 1, 3, 7, 15, 31, 63, 127, 255]
    i=int(i)
    code=0

    # hard encoding for integer 0
    if abs(i) == 0:
        code = 0

    # encoding of non-zero values
    else:
        for j in range(8):
            if  s[j] < abs(i) <= s[j+1] :
                # common bits at starting for positive and negative integers
                num= j+1
                code=num* [1]
                code.append(0)

                # for positive integers after appending ones and a zero we need to append its binary value as last some bits
                if i > 0:
                    code.append((bin(abs(i))[2:]))

                # for negative integers after appending ones and a zero we need to append its complement of binary value as last some bits
                # .rjust ensures num of bits are "j+1"
                else:
                    code.append((str(bin(s[j + 1] + i)[2:]).rjust((j + 1), '0')))

                code=int(''.join(str(i) for i in code))

    #encoded_string=encoded_string+str(code)
    return  str(code)



im_path='cameraman.tif'
image=io.imread(im_path)
m, n = image.shape


# quantization table given question
q_table = np.array([[16,11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])


mse,compressed_size = JPEG_compress(image,q_table,view = True)

MSE = []
mul_factor = np.arange(0.1,2,0.2)
size = []
for factor in mul_factor:
    #print('\n\n')
    mse,compressed_size = JPEG_compress(image,q_table*factor, factor = factor,prints = False)
    MSE.append(mse)
    size.append(compressed_size)
    

plt.plot(MSE,size,'-o')
_ = [plt.annotate("{:.2f}".format(z), (x, y)) for x, y, z in zip(MSE, size, mul_factor)]
plt.xlabel('MSE')
plt.ylabel('Compressed_size')
plt.title('MSE vs Compressed_size')
plt.savefig("MSE_vs_Compressed_size.jpg")
plt.show()

################################################### Q1- part 2 #########################################################









