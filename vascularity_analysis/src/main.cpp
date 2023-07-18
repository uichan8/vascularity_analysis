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
    // ���� â ����

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_8U);

    // ��� �ȼ� ����
    mask.at<uchar>(50, 50) = 255;
    mask.at<uchar>(51, 51) = 255;
    mask.at<uchar>(52, 52) = 255;

    int x = 50;
    int y = 50;

    // Circle Ŭ���� �ʱ�ȭ
    Circle C(5);

    // �б� ����ũ ã��
    std::vector<std::vector<int>> branch_mask = find_branch_mask(mask, x, y, C);

    // ��� ���
    for (const auto& row : branch_mask) {
        for (int value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    return 0;

}