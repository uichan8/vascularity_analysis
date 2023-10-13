#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

#include "vascularity.hpp"
#include "skeletonize.hpp"

using namespace std;
using namespace cv;


int main() {
    // 경로 긁어오기
    string mask_path = "C:/Users/uicha/Desktop/vascularity_analysis_cpp/data/mask/017_mask.png";
    string img_path = "C:/Users/uicha/Desktop/vascularity_analysis_cpp/data/img/017_img.png";

    // 이미지 파일 읽어오기
    Mat mask = imread(mask_path);
    Mat image = imread(img_path);

    // 벡터 선언
    vascularity example(image, mask);

    example.visualize();


    return 0;
}

