#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

#include "vascularity.hpp"
#include "skeletonize.hpp"

using namespace std;
using namespace cv;


int main2() {
    // 경로 긁어오기
    string mask_path = "C:/Users/82109/Desktop/vascularity_analysis_cpp/data/mask/000_mask.png";
    string img_path = "C:/Users/82109/Desktop/vascularity_analysis_cpp/data/img/000_img.png";

    // 이미지 파일 읽어오기
    Mat mask = imread(mask_path);
    Mat image = imread(img_path);

    // 벡터 선언
    vascularity example(image,mask);


    return 0;
}

int main(void) {
    string mask_path = "C:/Users/82109/Desktop/vascularity_analysis_cpp/data/mask/000_mask.png";
    Mat mask = imread(mask_path);
    Mat mask_c[3];
    split(mask, mask_c);

    Mat skel;
    skeletonize(mask_c[0], skel);

    cv::imwrite("skel_result005.bmp", skel);
    return 0;
}
