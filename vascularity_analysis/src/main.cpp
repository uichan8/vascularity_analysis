#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>

#include "vascularity.hpp"
#include "branch_vectorization.hpp"
#include "bifur_vectorization.hpp"

using namespace std;
using namespace cv;


int main() {
    // ��� �ܾ����
    string mask_path = "C:/Users/82109/Desktop/vascularity_analysis_cpp/data/mask/000_mask.png";
    string img_path = "C:/Users/82109/Desktop/vascularity_analysis_cpp/data/img/000_img.png";

    // �̹��� ���� �о����
    Mat mask = imread(mask_path);
    Mat image = imread(img_path);

    // ���� ����
    vascularity example(image,mask);

    return 0;
}