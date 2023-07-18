#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>

#include "vessel_width_detection.hpp"
#include "bifur_vectorization.hpp"
#include "branch_vectorization.hpp"

#define M_PI 3.14159265358979323846

using namespace std;
using namespace cv;


int main() {
    // 검은 창 생성

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8U);

    // 흰색 픽셀 설정
    mask.at<uchar>(50, 50) = 255;
    mask.at<uchar>(51, 51) = 255;
    mask.at<uchar>(52, 52) = 255;

    int x = 50;
    int y = 50;

    // Circle 클래스 초기화
    Circle C(5);

    // 분기 마스크 찾기
    std::vector<std::vector<int>> branch_mask = find_branch_mask(mask, x, y, C);

    // 결과 출력
    for (const auto& row : branch_mask) {
        for (int value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    return 0;

}