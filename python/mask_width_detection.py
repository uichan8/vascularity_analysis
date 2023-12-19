import numpy as np

def mask_width_detection(seg_mask, pts_arr):

    x1_edge = np.array([])
    x2_edge = np.array([])
    y1_edge = np.array([])
    y2_edge = np.array([])

    for i in range(0, len(pts_arr)-1,2):
        if i == len(pts_arr)-1:
            (x, y) = (pts_arr[i][0], pts_arr[i][1])
            (diff_x, diff_y) = (pts_arr[i][0] - pts_arr[i-1][0], pts_arr[i][1] - pts_arr[i-1][1])
        else:       
            (x, y) = (pts_arr[i][0], pts_arr[i][1])
            (diff_x, diff_y) = (pts_arr[i+1][0] - pts_arr[i][0], pts_arr[i+1][1] - pts_arr[i][1])
            if diff_x == 0 and diff_y == 0:
                (diff_x, diff_y) = (pts_arr[i+2][0] - pts_arr[i][0], pts_arr[i+2][1] - pts_arr[i][1])
            (normal_x, normal_y) = (diff_y, -diff_x)
        
        x1 = x
        x2 = x
        y1 = y
        y2 = y
        while(x1<seg_mask.shape[1] and y1< seg_mask.shape[0]):
            x_prime = x1 + normal_x
            x_pprime = x2 - normal_x
            y_prime = y1 + normal_y
            y_pprime = y2 - normal_y

            # img[int(x_prime), int(y_prime)] = [255, 255, 255]
            # img[int(x_prime), int(y_pprime)] = [255, 255, 255]
            if seg_mask[int(y_prime), int(x_prime)] == 0 and seg_mask[int(y_pprime), int(x_pprime)] == 0:
                x1_edge = np.append(x1_edge,x_prime)
                x2_edge = np.append(x2_edge, x_pprime)
                y1_edge = np.append(y1_edge,y_prime)
                y2_edge = np.append(y2_edge,y_pprime)
                break
            if seg_mask[int(y_prime), int(x_prime)] == 0:
                x_prime = x1
                y_prime = y1
            if seg_mask[int(y_pprime), int(x_pprime)] == 0:
                x_pprime = x2
                y_pprime = y2

            x1 = x_prime
            x2 = x_pprime
            y1 = y_prime
            y2 = y_pprime 

    return x1_edge , x2_edge, y1_edge, y2_edge  

'''
def mask_width_detection(seg_mask, x, y , g):
    x1_edge = np.array([])
    x2_edge = np.array([])
    y1_edge = np.array([])
    y2_edge = np.array([])

    for i in range(len(x)):

        x1 = x[i]
        x2 = x[i]
        y1 = y[i]
        y2 = y[i]
        angle = g[i]

        if seg_mask[int(x1), int(y1)] == 0 and i != 0:
            pass
        else:
            r1 = 0.05
            r2 = -0.05


        while(x1 <seg_mask.shape[0] and y1 < seg_mask.shape[1]):
            x_prime = x1 + r1 * np.sin(angle)
            x_pprime = x2 + r2 *np.sin(angle)
            y_prime = y1 + r1 * np.cos(angle)
            y_pprime = y2 + r2 * np.cos(angle)

            # img[int(x_prime), int(y_prime)] = [255, 255, 255]
            # img[int(x_prime), int(y_pprime)] = [255, 255, 255]
            if seg_mask[int(x_prime), int(y_prime)] == 0 and seg_mask[int(x_pprime), int(y_pprime)] == 0:

                x1_edge = np.append(x1_edge,x_prime-0.5)
                x2_edge = np.append(x2_edge, x_pprime-0.5)
                y1_edge = np.append(y1_edge,y_prime-0.5)
                y2_edge = np.append(y2_edge,y_pprime-0.5)

                break
            if seg_mask[int(x_prime), int(y_prime)] == 0:
                x_prime = x1
                y_prime = y1
                r1 -= 0.05
            if seg_mask[int(x_pprime), int(y_pprime)] == 0:
                x_pprime = x2
                y_pprime = y2
                r2 += 0.05

            x1 = x_prime
            x2 = x_pprime
            y1 = y_prime
            y2 = y_pprime 
            r1 += 0.05
            r2 -= 0.05

    return x1_edge , x2_edge, y1_edge, y2_edge  
'''

