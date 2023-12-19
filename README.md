# vascularity_analysis_cpp
## 코드를 사용하기 위한 초기 설정
1. opencv 설치\
링크 : https://blog.naver.com/chandong83/222682092712
2. include, src 폴더 추가 포함 디렉터리에 추가

# Vessel Vectorization
## 개요
 본 코드는 serval 프로그램의 일부로, 안저 영상에서 혈관 mask를 찾고 이를 활용하여 벡터화 하는 코드입니다.

## 과정
1. 마스크를 skeletonize 하여 중심선을 추출합니다.

2. skeletonize 이미지에서 분기점(branch) 를 찾습니다.

3. 분기점과 분기점사이에 있는 픽셀들을 이용하여 Polynomial approximation를 진행하여 다항식을 찾습니다.

4. 분기점으로부터 일정한 거리를 잡습니다. 구간에서의 마스크 기반 혈관 길이를 구하고, 중간값을 구합니다. 이는 분기점에서 뻗터나간 혈관의 대략적인 굵기로, 이를 토대로하여 마스크에서 branch와 vertex를 구분해 냅니다.

5. 각각을 벡터로 만듭니다.  
 a. 분기점은 중심 좌표와 마스크의 모양 정보로 나타냅니다.  
 b. 혈관은 모양과 굵기를 각각을 다항식으로 나타냅니다. 혈관x 혈관y 굵기r 세가지 함수로 나타납니다.
 이때 subpixel edge swg 알고리즘을 통해 엄밀한 파라미터를 얻습니다.

 ## 기타구현 해야될 부분
  - 좌표주어 졌을때 가까운 centerpoint찾기 -> 이미지를 격자로 나눈후 그안에서 찾으면 될듯
