#ifndef _vascularity
#define _vascularity

#include <vector>
#include <string>

#include "opencv2/opencv.hpp"
#include "graph_structure.hpp"

class vascularity {
private:
    //images
    cv::Mat fundus;    // �ȱ� �̹���
    cv::Mat mask;      // �ȱ� ����ũ
    cv::Mat skel;      // ���̷��� �̹���
    cv::Mat fa;        // �ȱ� ���� ����(optional)

    //optic disk
    cv::Point optic_disk_center; //optic disk �߽� ��ǥ
    int optic_disk_r; // optic disk ������

    //vector ����
    vgraph a_graph; //���� �׷��� ���� r            
    vgraph v_graph; //���� �׷��� ���� b

    std::vector<vtree*> a_tree; //���� Ʈ�� ����                  
    std::vector<vtree*> v_tree; //���� Ʈ�� ����   

    //��Ÿ �ڷ�
    cv::Mat node_map;  // gui���� � ������� �������ٲ��� ���̵� ���ִ� ����ũ 

public:
    //������
    vascularity(cv::Mat img, cv::Mat vmask);

    //vecterization
    void make_graph();
    void simple_vectorization(); //���Ἲ�� ���� �� ����

    //��Ÿ �޼ҵ�
    void where(const cv::Mat& skel, std::vector<cv::Point> &result);
    bool write(std::string path = "result/vector_json/recent_result.json");
    void visualize(int sampling_dis = 1,bool save = false);
};

#endif