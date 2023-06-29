#include <opencv2/opencv.hpp>

int main() {
    // 검은 창 생성
    cv::Mat blackImage = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);

    // 윈도우 생성 및 이미지 표시
    cv::namedWindow("Black Image", cv::WINDOW_NORMAL);
    cv::imshow("Black Image", blackImage);

    // 키 입력 대기
    cv::waitKey(0);

    // 윈도우 닫기
    cv::destroyAllWindows();

    return 0;
}