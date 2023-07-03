#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>

#include "vessel_width_detection.hpp"
#include "bifur_vectorization.hpp"

#define M_PI 3.14159265358979323846

using namespace std;
using namespace cv;

int main() {
    // ���� â ����
    Circle C(19);

    cv::Mat mask = cv::Mat::zeros(100, 100, CV_32S);

    int x = 50;
    int y = 50;

    std::vector<std::vector<int>> branch_mask = find_branch_mask(mask, x, y, C);

    // branch_mask ���
    for (const auto& row : branch_mask) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    cv::Mat blackImage = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);

    // ������ ���� �� �̹��� ǥ��
    cv::namedWindow("Black Image", cv::WINDOW_NORMAL);
    cv::imshow("Black Image", blackImage);

    // Ű �Է� ���
    //cv::waitKey(0);

    // ������ �ݱ�
    cv::destroyAllWindows();





    return 0;
}