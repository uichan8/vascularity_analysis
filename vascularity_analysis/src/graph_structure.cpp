#include <iostream>
#include <algorithm>
#include <vector>

#include "graph_structure.hpp"
#include "opencv2/opencv.hpp"

using namespace std;

/*구조체*/

//-----------------------------------------------------------------------
//--------------------------   vnode   ----------------------------------
//-----------------------------------------------------------------------

/*getter, setter*/
vector<vnode*>& vnode::get_head() {
	return head;
}

vector<vnode*>& vnode::get_tail() {
	return tail;
}

/*method*/
void vnode::change_head_tail() {
	swap(head, tail);
}
void vnode::push_head(vnode* target) {
	head.push_back(target);
};

void vnode::push_tail(vnode* target) {
	tail.push_back(target);
};

void vnode::disconnect(vnode* target) {
	head.erase(remove(head.begin(), head.end(), target), head.end());
	tail.erase(remove(tail.begin(), tail.end(), target), tail.end());
};


//-----------------------------------------------------------------------
//--------------------------   vbranch   --------------------------------
//-----------------------------------------------------------------------

/*getter, setter*/
vector<int*>& vbranch::get_branch_segments() {
	return branch_segments;
}

void vbranch::set_branch_segments(const vector<int*>& segments) {
	branch_segments = segments;
}

vector<int*>& vbranch::get_branch_r() {
	return branch_r;
}

void vbranch::set_branch_r(const vector<int*>& rValues) {
	branch_r = rValues;
}

/*method*/
void vbranch::add_segment_data(int* segment, int* r) {
	branch_segments.push_back(segment);
	branch_r.push_back(r);
}

//-----------------------------------------------------------------------
//-------------------------    vbifur    --------------------------------
//-----------------------------------------------------------------------

/*getter, setter*/
cv::Point& vbifur::get_center_coor() {
	return center_coor;
}

cv::Mat& vbifur::get_vbifur_mask() {
	return vbifur_mask;
}

void vbifur::set_center_coor(cv::Point& center) {
	center_coor = center;
}

void vbifur::set_vbifur_mask(cv::Mat& mask) {
	vbifur_mask = mask;
}

//-----------------------------------------------------------------------
//-------------------------     graph     -------------------------------
//-----------------------------------------------------------------------

void vgraph::add_bifur(vbifur new_bifur) {
	vbifurs.push_back(new_bifur);
}

void vgraph::add_branch(vbranch new_branch) {
	vbranches.push_back(new_branch);
}

void vgraph::connect() {
	//1. bifur을 이미지를 나타내는데, 1번 bifur에 해당하는 마스크는 픽셀값을 1로 둔다.
	//2. 다이레이트를 한다.
	//3. branch의 end point의 픽셀값이랑 연결해준다.
}