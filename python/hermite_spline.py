import numpy as np

def hermite_spline(x1,y1,g1,x2,y2,g2):
    A = np.array([
        [x1**3,x1**2,x1,1],
        [x2**3,x2**2,x2,1],
        [3*x1**2,2*x1,1,0],
        [3*x2**2,2*x2,1,0]
    ])

    B = np.array([y1,y2,g1,g2])
    return np.linalg.solve(A,B)

def substitute(coefficient,x):
    return coefficient[0]*x**3 + coefficient[1]*x**2 + coefficient[2]*x + coefficient[3]

def fit(x, y, k = 1):
    """
        k 는 기울기 강도 커질수록 언더피팅을 유발함
    """
    if len(x) < 5:
        return [], []
    x = list(x)
    y = list(y)
    xi = np.array([x[0]] + x + [x[-1]])
    yi = np.array([y[0]] + y + [y[-1]])
    poly_x = []
    poly_y = []
    gx = (xi[2:] - xi[:-2])/k
    gy = (yi[2:] - yi[:-2])/k
    for i in range(len(x)-1):
        result_x = hermite_spline(0,x[i],gx[i],1,x[i+1],gx[i+1])
        result_y = hermite_spline(0,y[i],gy[i],1,y[i+1],gy[i+1])

        poly_x.append(result_x)
        poly_y.append(result_y)

    return poly_x, poly_y

def get_lines(poly_x,sample_num = 10):
    sample = np.linspace(0,1,sample_num,endpoint= False)
    result_x = []

    for i in range(len(poly_x)):
        x = substitute(poly_x[i],sample)
        result_x += list(x)

    return np.array(result_x)

def differentiate(poly_array):
    diff_array = []
    for poly in poly_array:
        diff_array.append(np.array([0,3*poly[0],2*poly[1],poly[2]]))

    return diff_array
        


if __name__ == "__main__":
    a = hermite_spline(1,2,0.5,3,4,0.5)
    print(a)
    print(substitute(a, 1))
    print(substitute(a, 3))
    x = np.array([1, 2, 3, 4, 5, 6, 7])
    y = np.array([2, 4, 1, 5, 3, 2, 3])
    

    k = 1.5
    poly = fit(x, y, k)

    sample_num = 2
    lines_x = get_lines(poly[0], sample_num)
    lines_y = get_lines(poly[1], sample_num)
    diff_x = differentiate(poly[0])
    diff_y = differentiate(poly[1])
    
    print(diff_x)
    print(diff_y)