# def sampling_points(sorted_x, sorted_y):
#     start_point = (sorted_x[0],sorted_y[0])
#     gradient = (sorted_x[1]-sorted_x[0],sorted_y[1]-sorted_y[0])

#     sample_x = [sorted_x[0]]
#     sample_y = [sorted_y[0]]
#     sample_g = [gradient]
    
#     start_point = (sorted_x[0],sorted_y[0])
#     for i in range(1,len(sorted_x)-1):
#         gradient1 = (sorted_x[i]-sorted_x[i-1],sorted_y[i]-sorted_y[i-1])
#         gradient2 = (sorted_x[i+1]-sorted_x[i],sorted_y[i+1]-sorted_y[i])
#         if gradient1 != gradient2:
#             end_point = (sorted_x[i], sorted_y[i])
#             center_point = ((start_point[0]+end_point[0])/2,(start_point[1]+end_point[1])/2)
#             start_point = end_point
#             sample_x.append(center_point[0])
#             sample_y.append(center_point[1])
#     sample_x.append(sorted_x[-1])
#     sample_y.append(sorted_y[-1])
            
#     return sample_x, sample_y
import numpy as np


def simple_sampling(arr, sparsity):
    sample_arr = []
    for i in range(len(arr)):
        if i % sparsity == 0:
            sample_arr.append(arr[i])
    arr = sample_arr

    return np.array(arr)

def delete_outliers(x_data, y_data, r_data, diff_data , threshhold = 3):
    mean = sum(r_data) / len(r_data)
    variance = sum((x - mean) ** 2 for x in r_data)/ len(r_data)
    std_dev = np.sqrt(variance)
    r_copy = r_data.copy()
    count = 0

    for i in range(len(r_copy)) :
        score = (r_copy[i] - mean) / std_dev
        if abs(score) > threshhold:
            x_data = np.delete(x_data, i-count)
            y_data = np.delete(y_data, i-count)
            r_data = np.delete(r_data, i-count)
            diff_data = np.delete(diff_data, i-count)
            count += 1

    return x_data, y_data, r_data, diff_data

