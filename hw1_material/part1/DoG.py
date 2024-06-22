import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        base_img = image
        for i in range(self.num_octaves):
            gaussian_images.append(base_img)
            for j in range(1, self.num_guassian_images_per_octave):
                gaussian_images.append(cv2.GaussianBlur(base_img, (0,0), self.sigma**(j)))
            if (i==0):
                base_img = cv2.resize(gaussian_images[-1], (base_img.shape[1]//2, base_img.shape[0]//2),
                                        interpolation = cv2.INTER_NEAREST)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        sub_run = int(len(gaussian_images)/2)
        for i in range(self.num_octaves):
            for j in range(1, sub_run):
                diff = cv2.subtract(gaussian_images[i*sub_run+j], gaussian_images[i*sub_run+j-1])
                dog_images.append(diff)
                # M, m = max(diff.flatten()), min(diff.flatten())
                # normalize = (diff-m)*255/(M-m)
                # cv2.imwrite(f'testdata/DoG{i+1}-{j}.png', normalize)
        
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        pos = []
        for run in range(self.num_octaves):
            arr = np.array(dog_images[:4]) if run == 0 else np.array(dog_images[4:])
            for i, j, k in np.ndindex(arr.shape[:]):
                if i > 0 and i < arr.shape[0]-1 \
                    and j > 0 and j < arr.shape[1]-1 \
                    and k > 0 and k < arr.shape[2]-1:
                    neighbors = arr[i-1:i+2, j-1:j+2, k-1:k+2]
                    max_neighbor, min_neighbor = np.max(neighbors), np.min(neighbors)
                    if (arr[i][j][k] == max_neighbor or arr[i][j][k] == min_neighbor) \
                        and np.abs(arr[i][j][k]) > self.threshold:
                        pos.append([j, k] if run == 0 else [2*j, 2*k])

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.array(pos)
        keypoints = np.unique(keypoints, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        return keypoints