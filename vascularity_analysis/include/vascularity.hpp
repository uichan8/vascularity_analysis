#ifndef _vascularity
#define _vascularity

#include <vector>

#include "opencv2/opencv.hpp"
#include "graph_structure.hpp"

class vascularity {
private:
    //images
    cv::Mat fundus;    // 안구 이미지
    cv::Mat mask;      // 안구 마스크
    cv::Mat skel;      // 스켈레톤 이미지
    cv::Mat fa;        // 안구 조영 사진(optional)

    //optic disk
    cv::Point optic_disk_center; //optic disk 중심 좌표
    int optic_disk_r; // optic disk 반지름

    //vector 구조
    std::vector<vgraph*> a_graph; //동맥 그래프 구조                  
    std::vector<vgraph*> v_graph; //정맥 그래프 구조

    std::vector<vtree*> a_tree; //동맥 트리 구조                  
    std::vector<vtree*> v_tree; //정맥 트리 구조   

    //기타 자료`    
    cv::Mat node_map;  // gui에서 어떤 대상으로 매핑해줄껀지 가이드 해주는 마스크 

public:
    //생성자
    vascularity(cv::Mat img, cv::Mat vmask);

    //그래프 구척 관련 메소드
    void make_graph();

    void skeletonize(const cv::Mat& src, cv::Mat& dst);
    void skel_iteration(cv::Mat& img, int iter);

    void branch_mask_split(const cv::Mat& skel, cv::Mat& branch_map, cv::Mat& bifur_map);

};

#endif