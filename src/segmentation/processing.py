import cv2
import numpy as np


#TODO split the processing into subfonction + provide a list of processor

class LabyProcessing(object):

    def __init__(self, threshold,
                 apply_denoising=False, denoising_factor=50,
                 apply_erode_dilate=False, kernel=5, no_erode=1,
                 apply_gaussblur=False, gauss_kernel=5, gauss_std=1):

        self.threshold = threshold

        self.apply_denoising = apply_denoising
        self.denoising_factor = denoising_factor

        self.apply_erode_dilate = apply_erode_dilate
        self.kernel = (kernel,kernel)
        self.no_erode = no_erode

        self.apply_gaussblur = apply_gaussblur
        self.gauss_kernel = (gauss_kernel,gauss_kernel)
        self.gauss_std = gauss_std

        assert kernel % 2 == 1 and gauss_kernel % 2 == 1

    def process(self, img):

        if self.apply_denoising:
            img = cv2.fastNlMeansDenoising(img, None, self.denoising_factor, 7, 21)

        if self.apply_gaussblur:
            img = cv2.GaussianBlur(img, self.gauss_kernel, sigmaX=self.gauss_std, sigmaY=self.gauss_std)

        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        fshift[:(crow - 90), :] = 0
        fshift[(crow + 90):, :] = 0
        fshift[:, (ccol + 90):] = 0
        fshift[:, :(ccol - 90):] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.abs(np.fft.ifft2(f_ishift))

        wimg = img_back.copy()

        # Seuillage
        wimg[np.where(wimg > self.threshold)] = 0
        wimg[np.where(wimg <= self.threshold)] = 255

        if self.apply_erode_dilate:
            wimg = cv2.erode(wimg, np.ones(self.kernel, np.uint8), iterations=self.no_erode)
            wimg = cv2.dilate(wimg, np.ones(self.kernel, np.uint8), iterations=self.no_erode)

        return wimg

