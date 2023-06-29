#include <opencv2/opencv.hpp>

int main() {
    // ���� â ����
    cv::Mat blackImage = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);

    // ������ ���� �� �̹��� ǥ��
    cv::namedWindow("Black Image", cv::WINDOW_NORMAL);
    cv::imshow("Black Image", blackImage);

    // Ű �Է� ���
    cv::waitKey(0);

    // ������ �ݱ�
    cv::destroyAllWindows();

    return 0;
}