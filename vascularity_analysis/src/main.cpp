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
    cv::Mat image(300, 400, CV_8UC3, cv::Scalar(255, 255, 255));

    // ���� �׸��� ���� �������� ���� ��ǥ
    cv::Point point1(50, 100);
    cv::Point point2(350, 200);

    // �� �׸���
    draw_line(image, point1, point2, 'r', 2);

    // �̹��� ���
    cv::imshow("Line Drawing Example", image);
    cv::waitKey(0);

    return 0;

}