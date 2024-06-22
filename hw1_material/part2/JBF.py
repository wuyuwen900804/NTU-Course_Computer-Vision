import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        # Spatial kernel
        Gs = np.exp(-0.5*(np.arange(self.pad_w+1)**2)/self.sigma_s**2)
        # Range kernel
        Gr = np.exp(-0.5*(np.arange(256)/255)**2/self.sigma_r**2)

        # Compute intensity of pixel p of filtered image I'
        weight = np.zeros_like(padded_img, dtype=np.float64)
        result = np.zeros_like(padded_img, dtype=np.float64)
        for x in range(-self.pad_w, self.pad_w+1):
            for y in range(-self.pad_w, self.pad_w+1):
                # Gs(p,q)
                sw = Gs[np.abs(x)]*Gs[np.abs(y)]
                # Gr(p,q)
                diff = Gr[np.abs(np.roll(padded_guidance, [y,x], axis=[0,1])-padded_guidance)]
                rw = diff if diff.ndim==2 else np.prod(diff,axis=2)
                # Gs(p,q) * Gr(p,q)
                tw = sw*rw
                padded_img_roll = np.roll(padded_img, [y,x], axis=[0,1])
                for channel in range(padded_img.ndim):
                    # Sum Gs(p,q) * Gr(p,q) * Iq for all q
                    result[:,:,channel] += np.multiply(padded_img_roll[:,:,channel], tw)
                    # Sum Gs(p,q) * Gr(p,q) for all q
                    weight[:,:,channel] += tw

        # [Sum Gs(p,q) * Gr(p,q) * Ip] / [Sum Gs(p,q) * Gr(p,q) for all q]
        result, weight = np.array(result), np.array(weight)
        output = np.divide(result, weight)[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w,:]
        return np.clip(output, 0, 255).astype(np.uint8)