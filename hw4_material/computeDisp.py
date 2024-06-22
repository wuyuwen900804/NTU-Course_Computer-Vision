import numpy as np
import cv2
import cv2.ximgproc as xip

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    
    # add padding
    imgL = cv2.copyMakeBorder(Il, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    imgR = cv2.copyMakeBorder(Ir, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # matrix to record 9 binary values
    imgL_bin = np.zeros((9, h+2, w+2, ch))
    imgR_bin = np.zeros((9, h+2, w+2, ch))

    # roll the image with 3x3 window
    # # # # # # # # # # # # # # # # # # # #
    #            |           |            #
    #   idx = 0  |  idx = 3  |  idx = 6   #
    # ___________|___________|___________ #
    #            |           |            #
    #   idx = 1  |  idx = 4  |  idx = 7   #
    # ___________|___________|___________ #
    #            |           |            #
    #   idx = 2  |  idx = 5  |  idx = 8   #
    #            |           |            #
    # # # # # # # # # # # # # # # # # # # #
    idx = 0
    for x in range(-1, 2):
        for y in range(-1, 2):
            # assign "1" if the value of central pixel is greater than peripheral pixel
            maskL = (imgL > np.roll(imgL, [y, x], axis=[0, 1]))
            imgL_bin[idx][maskL] = 1
            maskR = (imgR > np.roll(imgR, [y, x], axis=[0, 1]))
            imgR_bin[idx][maskR] = 1
            idx += 1

    # remove the padded
    imgL_bin = imgL_bin[:, 1:-1, 1:-1, :]
    imgR_bin = imgR_bin[:, 1:-1, 1:-1, :]

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)

    l_cost_record = np.zeros((max_disp+1, h, w))
    r_cost_record = np.zeros((max_disp+1, h, w))
    wndw_size = -1 # calculate window size from spatial kernel
    sigma_r, sigma_s = 4, 12
    for d in range(max_disp+1):
        l_shift = imgL_bin[:, :, d:].astype(np.uint32)
        r_shift = imgR_bin[:, :, :w-d].astype(np.uint32)
        # compute Hamming distance with XOR(^).
        cost = np.sum(l_shift^r_shift, axis=0)
        cost = np.sum(cost, axis=2).astype(np.float32)
        # left-to-right
        l_cost = cv2.copyMakeBorder(cost, 0, 0, d, 0, cv2.BORDER_REPLICATE)
        l_cost_record[d] = xip.jointBilateralFilter(Il, l_cost, wndw_size, sigma_r, sigma_s)
        # right-to-left
        r_cost = cv2.copyMakeBorder(cost, 0, 0, 0, d, cv2.BORDER_REPLICATE)
        r_cost_record[d] = xip.jointBilateralFilter(Ir, r_cost, wndw_size, sigma_r, sigma_s)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all

    l_disp_map = np.argmin(l_cost_record, axis=0)
    r_disp_map = np.argmin(r_cost_record, axis=0)

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering

    # Left-right consistency check
    lr_check = np.zeros((h, w), dtype=np.float32)
    x, y = np.meshgrid(range(w),range(h))
    R_x = (x - l_disp_map) # x-DL(x,y)
    mask1 = (R_x >= 0)
    l_disp = l_disp_map[mask1]
    r_disp = r_disp_map[y[mask1], R_x[mask1]]
    mask2 = (l_disp == r_disp) # if DL(x,y) = DR(x-DL(x,y), y)
    lr_check[y[mask1][mask2], x[mask1][mask2]] = l_disp_map[mask1][mask2]

    # Hole filling
    lr_check_pad = cv2.copyMakeBorder(lr_check, 0, 0, 1, 1, cv2.BORDER_CONSTANT, value=max_disp)
    l_labels = np.zeros((h, w), dtype=np.float32)
    r_labels = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            idx_L, idx_R = 0, 0
            # FL, the disparity map filled by closest valid disparity from left
            while lr_check_pad[y, x - idx_L + 1] == 0:
                idx_L += 1
            l_labels[y, x] = lr_check_pad[y, x - idx_L + 1]
            # FR, the disparity map filled by closest valid disparity from right
            while lr_check_pad[y, x + idx_R + 1] == 0:
                idx_R += 1
            r_labels[y, x] = lr_check_pad[y, x + idx_R + 1]
    # Final filled disparity map D = min(FL, FR) (pixel-wise minimum)
    labels = np.min((l_labels, r_labels), axis=0)

    # Weighted median filtering
    radius = 15
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels, radius)

    return labels.astype(np.uint8)