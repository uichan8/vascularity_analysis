#ifndef _graph_structure
#define _graph_structure

#include "opencv2/opencv.hpp"
#include <vector>

//-----------------------------------------------------------------------
//--------------------------   ����ü   ---------------------------------
//-----------------------------------------------------------------------

class vnode {
protected:
	int ID_num;
	std::vector<vnode*> head, tail;

public:
	//getter
	int get_ID();
	std::vector<vnode*>& get_head();
	std::vector<vnode*>& get_tail();

	void set_ID(int num);

	//method
	void change_head_tail();		//Ʈ�� �������� ����� ���⼺�� �ٲٴ� �Լ� head���� optical_disk�� ����� ��
	void push_head(vnode* target);	//head�ʿ� ��� ��带 ����
	void push_tail(vnode* target);	//tail�ʿ� ��� ��带 ���� -> graph Ŭ�������� �ΰ��� ��ġ�� �޼ҵ� ���� �ؾ���
	void disconnect(vnode* parent, vnode* child); //�ش� ��尡 Ư�� ���� ����Ǿ� ���� �� ��� ������ ������ �����ϴ� �޼ҵ�
};

class vbranch : public vnode {
private:
	std::vector<std::vector<double>> branch_poly_x;
	std::vector<std::vector<double>> branch_poly_y;
	std::vector<std::vector<double>> branch_poly_r;
	cv::Point end_points[2]; //�׷������� �����۶� ���� �� �� �ʿ��� �ڵ�

public:
	//getter
	std::vector<std::vector<double>> get_poly_x();
	std::vector<std::vector<double>> get_poly_y();
	std::vector<std::vector<double>> get_poly_r();

	//setter
	void set_poly_x(const std::vector<std::vector<double>>& poly_x);
	void set_poly_y(const std::vector<std::vector<double>>& poly_y);
	void set_poly_r(const std::vector<std::vector<double>>& poly_r);
	void set_end_points(cv::Point pt1, cv::Point pt2);
};

class vbifur : public vnode {
private:
	cv::Point center_coor;
	cv::Mat vbifur_mask;
	std::vector<cv::Point> bifur_edge;

public:
	//getter
	cv::Point& get_center_coor();
	cv::Mat& get_vbifur_mask();
	std::vector<cv::Point>& get_bifur_edge();

	//setter
	void set_center_coor(cv::Point& center);
	void set_vbifur_mask(cv::Mat& mask);
	void set_bifur_edge(std::vector<cv::Point> edge);
};

//-----------------------------------------------------------------------
//--------------------------    ����    ---------------------------------
//-----------------------------------------------------------------------
class vgraph {
private:
	std::vector <vbranch> vbranches;
	std::vector <vbifur> vbifurs;
public:
	std::vector<vbranch> get_branch();
	std::vector<vbifur> get_bifur();
	void add_bifur(vbifur new_bifur);
	void add_branch(vbranch new_branch);
	vbifur& find_bifur(cv::Point center); //�߽� ��ǥ�� �Է��ϸ�, �ش��ϴ� bifur�� �������� �Լ�
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