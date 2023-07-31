#ifndef _graph_structure
#define _graph_structure

#include "opencv2/opencv.hpp"
#include <vector>

//-----------------------------------------------------------------------
//--------------------------   ����ü   ---------------------------------
//-----------------------------------------------------------------------

class vnode {
protected:
	std::vector<vnode*> head, tail;

public:
	//getter
	std::vector<vnode*>& get_head();
	std::vector<vnode*>& get_tail();

	//method
	void change_head_tail();		//Ʈ�� �������� ����� ���⼺�� �ٲٴ� �Լ� head���� optical_disk�� ����� ��
	void push_head(vnode* target);	//head�ʿ� ��� ��带 ����
	void push_tail(vnode* target);	//tail�ʿ� ��� ��带 ���� -> graph Ŭ�������� �ΰ��� ��ġ�� �޼ҵ� ���� �ؾ���
	void disconnect(vnode* target); //�ش� ��尡 Ư�� ���� ����Ǿ� ���� �� ��� ������ ������ �����ϴ� �޼ҵ�
};

class vbranch : public vnode {
private:
	std::vector<int*> branch_segments;
	std::vector<int*> branch_r;
	cv::Point end_points[2]; //�׷������� �����۶� ���� �� �� �ʿ��� �ڵ�

public:
	//getter
	std::vector<int*>& get_branch_segments();
	std::vector<int*>& get_branch_r();

	//setter
	void set_branch_segments(const std::vector<int*>& segments);
	void set_branch_r(const std::vector<int*>& rValues);

	//method
	void add_segment_data(int* segment, int* r); // branch���� ���� �κ����� ���� �Ŀ� ������ �����͸� �߰����ִ� �޼ҵ�
};

class vbifur : public vnode {
private:
	cv::Point center_coor;
	cv::Mat vbifur_mask;

public:
	//getter
	cv::Point& get_center_coor();
	cv::Mat& get_vbifur_mask();

	//setter
	void set_center_coor(cv::Point& center);
	void set_vbifur_mask(cv::Mat& mask);
};

//-----------------------------------------------------------------------
//--------------------------    ����    ---------------------------------
//-----------------------------------------------------------------------
class vgraph {
private:
	std::vector <vbranch> vbranches;
	std::vector <vbifur> vbifurs;
public:
	void add_bifur(vbifur new_bifur);
	void add_branch(vbranch new_branch);
	void connect();
};

// ���߿� �׷��� �ϼ� �� ���� �۾�
class vtree {
private:
	vnode* root;
	std::vector <vbranch> vbranches;
	std::vector <vbifur> vbifurs;
public:
};

#endif