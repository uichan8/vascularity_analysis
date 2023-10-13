#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

#include "vascularity.hpp"
#include "skeletonize.hpp"

using namespace std;
using namespace cv;


int main() {
    // ��� �ܾ����
    string mask_path = "C:/Users/uicha/Desktop/vascularity_analysis_cpp/data/mask/017_mask.png";
    string img_path = "C:/Users/uicha/Desktop/vascularity_analysis_cpp/data/img/017_img.png";

    // �̹��� ���� �о����
    Mat mask = imread(mask_path);
    Mat image = imread(img_path);

    // ���� ����
    vascularity example(image, mask);

    example.visualize();


    return 0;
}

