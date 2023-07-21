#ifndef _vascularity
#define _vascularity

#include <vector>

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
    std::vector<vgraph*> a_graph; //���� �׷��� ����                  
    std::vector<vgraph*> v_graph; //���� �׷��� ����

    std::vector<vtree*> a_tree; //���� Ʈ�� ����                  
    std::vector<vtree*> v_tree; //���� Ʈ�� ����   

    //��Ÿ �ڷ�`    
    cv::Mat node_map;  // gui���� � ������� �������ٲ��� ���̵� ���ִ� ����ũ 

public:
    //������
    vascularity(cv::Mat img, cv::Mat vmask);

    //�׷��� ��ô ���� �޼ҵ�
    void make_graph();

    void skeletonize(const cv::Mat& src, cv::Mat& dst);
    void skel_iteration(cv::Mat& img, int iter);

    void branch_mask_split(const cv::Mat& skel, cv::Mat& branch_map, cv::Mat& bifur_map);

};

#endif