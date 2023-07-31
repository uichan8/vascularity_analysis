#ifndef _graph_structure
#define _graph_structure

#include "opencv2/opencv.hpp"
#include <vector>

//-----------------------------------------------------------------------
//--------------------------   구조체   ---------------------------------
//-----------------------------------------------------------------------

class vnode {
protected:
	std::vector<vnode*> head, tail;

public:
	//getter
	std::vector<vnode*>& get_head();
	std::vector<vnode*>& get_tail();

	//method
	void change_head_tail();		//트리 구조에서 노드의 방향성을 바꾸는 함수 head쪽이 optical_disk와 가까운 쪽
	void push_head(vnode* target);	//head쪽에 대상 노드를 연결
	void push_tail(vnode* target);	//tail쪽에 대상 노드를 연결 -> graph 클래스에서 두개를 합치는 메소드 생성 해야함
	void disconnect(vnode* target); //해당 노드가 특정 노드와 연결되어 있을 때 대상 노드와의 연결을 해제하는 메소드
};

class vbranch : public vnode {
private:
	std::vector<int*> branch_segments;
	std::vector<int*> branch_r;
	cv::Point end_points[2]; //그래프에서 바이퍼랑 연결 할 때 필요한 코드

public:
	//getter
	std::vector<int*>& get_branch_segments();
	std::vector<int*>& get_branch_r();

	//setter
	void set_branch_segments(const std::vector<int*>& segments);
	void set_branch_r(const std::vector<int*>& rValues);

	//method
	void add_segment_data(int* segment, int* r); // branch에서 여러 부분으로 나눈 후에 각각의 데이터를 추가해주는 메소드
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
//--------------------------    구조    ---------------------------------
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

// 나중에 그래프 완성 후 수정 작업
class vtree {
private:
	vnode* root;
	std::vector <vbranch> vbranches;
	std::vector <vbifur> vbifurs;
public:
};

#endif