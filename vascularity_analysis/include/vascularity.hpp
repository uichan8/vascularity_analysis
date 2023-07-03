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
    cv::Mat fa;        // 안구 조영 사진(optional)

    //optic disk
    cv::Point optic_disk_center; //optic disk 중심 좌표
    int optic_disk_r; // optic disk 반지름

    //vector 구조
    std::vector<vgraph*> a_graph; //동맥 그래프 구조                  
    std::vector<vgraph*> v_graph; //정맥 그래프 구조

    std::vector<vtree*> a_tree; //동맥 트리 구조                  
    std::vector<vtree*> v_tree; //정맥 트리 구조   

    //기타 자료
    cv::Mat node_map;  // gui에서 어떤 대상으로 매핑해줄껀지 가이드 해주는 마스크 

public:
    vascularity(cv::Mat fundus, cv::Mat mask);
    ~vascularity();

};

#endif