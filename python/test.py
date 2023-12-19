import numpy as np
import find_branch
import sort_points
import matplotlib.pyplot as plt

mask = np.zeros((100, 100), dtype=np.uint8)

# 흰색 픽셀 설정
mask[50, 50] = 255
mask[51, 51] = 255
mask[52, 52] = 255

x = 50
y = 50

# Circle 클래스 초기화

# 분기 마스크 찾기
branch_mask = find_branch.find_branch_mask(mask, x, y)

# 결과 출력
for row in branch_mask:
    print(' '.join(str(value) for value in row))
