
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from numba import jit
from multiprocessing import Pool
import os
import glob
import matplotlib.pyplot as plt


def getImage(index, grayscale = False, scale = 0.5):
  '''
  Helper function that returns images given a certain image index
  '''
  if grayscale:
    grayscale = 0
  else:
    grayscale = 1
  gt = cv2.imread('data/Image' + str(index) + '.png', grayscale)
  gt = cv2.resize(gt, (0,0), fx = scale, fy = scale)
  return gt

def read_data(file_dir, fname, i):
  # fname_tmp = file_dir + "{:04}".format(i) + fname
  fname_tmp = file_dir + i + fname
  # print(fname_tmp)
  data = np.load(fname_tmp)
  return data

def enlarge(image):
  width = image.shape[1]*4
  height = image.shape[0]*4
  image_enlarged = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
  return image_enlarged


@jit(nopython=True)
def nonLocalMeans(noisy, params=tuple(), verbose=True):
  '''
  Performs the non-local-means algorithm given a noisy image.
  params is a tuple with:
  params = (bigWindowSize, smallWindowSize, h)
  Please keep bigWindowSize and smallWindowSize as even numbers
  '''

  bigWindowSize, smallWindowSize, h = params
  padwidth = bigWindowSize // 2
  image = noisy.copy()

  # The next few lines creates a padded image that reflects the border so that the big window can be accomodated through the loop
  paddedImage = np.zeros((image.shape[0] + bigWindowSize, image.shape[1] + bigWindowSize))
  paddedImage = paddedImage.astype(np.uint8)
  paddedImage[padwidth:padwidth + image.shape[0], padwidth:padwidth + image.shape[1]] = image
  paddedImage[padwidth:padwidth + image.shape[0], 0:padwidth] = np.fliplr(image[:, 0:padwidth])
  paddedImage[padwidth:padwidth + image.shape[0], image.shape[1] + padwidth:image.shape[1] + 2 * padwidth] = np.fliplr(
    image[:, image.shape[1] - padwidth:image.shape[1]])
  paddedImage[0:padwidth, :] = np.flipud(paddedImage[padwidth:2 * padwidth, :])
  paddedImage[padwidth + image.shape[0]:2 * padwidth + image.shape[0], :] = np.flipud(
    paddedImage[paddedImage.shape[0] - 2 * padwidth:paddedImage.shape[0] - padwidth, :])

  iterator = 0
  totalIterations = image.shape[1] * image.shape[0] * (bigWindowSize - smallWindowSize) ** 2

  if verbose:
    print("TOTAL ITERATIONS = ", totalIterations)

  outputImage = paddedImage.copy()

  smallhalfwidth = smallWindowSize // 2

  # For each pixel in the actual image, find a area around the pixel that needs to be compared
  for imageX in range(padwidth, padwidth + image.shape[1]):
    for imageY in range(padwidth, padwidth + image.shape[0]):

      bWinX = imageX - padwidth
      bWinY = imageY - padwidth

      # comparison neighbourhood
      compNbhd = paddedImage[imageY - smallhalfwidth:imageY + smallhalfwidth + 1,
                 imageX - smallhalfwidth:imageX + smallhalfwidth + 1]

      pixelColor = 0
      totalWeight = 0

      # For each comparison neighbourhood, search for all small windows within a large box, and compute their weights
      for sWinX in range(bWinX, bWinX + bigWindowSize - smallWindowSize, 1):
        for sWinY in range(bWinY, bWinY + bigWindowSize - smallWindowSize, 1):
          # find the small box
          smallNbhd = paddedImage[sWinY:sWinY + smallWindowSize + 1, sWinX:sWinX + smallWindowSize + 1]
          euclideanDistance = np.sqrt(np.sum(np.square(smallNbhd - compNbhd)))
          # weight is computed as a weighted softmax over the euclidean distances
          weight = np.exp(-euclideanDistance / h)
          totalWeight += weight
          pixelColor += weight * paddedImage[sWinY + smallhalfwidth, sWinX + smallhalfwidth]
          iterator += 1

          if verbose:
            percentComplete = iterator * 100 / totalIterations
            if percentComplete % 5 == 0:
              print('% COMPLETE = ', percentComplete)

      pixelColor /= totalWeight
      outputImage[imageY, imageX] = pixelColor

  return outputImage[padwidth:padwidth + image.shape[0], padwidth:padwidth + image.shape[1]]

def denoise_gaussian(image, kernel):
  gFilteredGNoised = cv2.GaussianBlur(image, kernel, 0)
  return gFilteredGNoised

def denoise_nlm(image, gParams):
  nlmFilteredGNoised = nonLocalMeans(image, params=(gParams['bigWindow'], gParams['smallWindow'], gParams['h']),
                                     verbose=False)
  return nlmFilteredGNoised

if __name__ == "__main__":
  file_dir = './data/Train/'
  fname_noisyDWIk = '_NoisyDWIk.npy'
  fname_gt = '_gtDWIs.npy'

  dir_list = glob.glob(os.path.join(file_dir, '*_NoisyDWIk.npy'))
  dir_list = [os.path.basename(file_name) for file_name in dir_list]
  id = [file_name.split('_')[0] for file_name in dir_list]

  kernelSize = 3
  kernel = (kernelSize, kernelSize)

  gParams = {
    'bigWindow': 20,
    'smallWindow': 6,
    'h': 14,
    'scale': 2,
  }

  for index in id:
    noise = read_data(file_dir, fname_noisyDWIk, index)
    noise = np.abs(np.fft.ifft2(noise, axes=(0, 1), norm='ortho'))*255

    gt = read_data(file_dir, fname_gt, index)
    gt = abs(gt)

    stake = []
    for layer in range(noise.shape[2]):
      noise_slides = noise[:,:,layer]
      noise_slides_de = denoise_gaussian(noise_slides, kernel)
      # noise_slides_de = denoise_nlm(noise_slides, gParams)
      stake.append(noise_slides_de)
    result = np.stack(stake, axis=2)
    np.save('./res_nlm/' + index + fname_noisyDWIk, result)
    
    # plt.subplot(1,4,1)
    # plt.imshow(noise[:,:,2])
    # plt.title('noised')

    # plt.subplot(1,4,2)
    # plt.imshow(result[:,:,2])
    # plt.title('denoised')

    # plt.subplot(1,4,3)
    # plt.imshow(gt[:,:,2])
    # plt.title('Origin')

    # plt.subplot(1,4,4)
    # bias = abs(gt[:,:,2]*255 - noise[:,:,2])
    # print("average noise", index ," = ",np.mean(bias))
    # plt.imshow(bias)#, vmin=70, vmax=100
    # plt.title('noise - gt')

    # plt.show()


  # cv2.imshow('noise', enlarge(noise))
  # cv2.waitKey(0)
  #
  # gt = read_data(file_dir, fname_gt, '0001')
  # gt = gt[:, :, 4]
  # gt = abs(gt)
  # cv2.imshow('ground truth', enlarge(gt))
  # cv2.waitKey(0)
  #
  # # Parameters for denoising using gaussian filter
  # kernelSize = 3
  # kernel = (kernelSize, kernelSize)
  # noise_de_gau =  denoise_gaussian(noise, kernel)
