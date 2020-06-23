#include "anchor_generator.h"
#include "detect.h"
#include "tools.h"
#include<iostream>


using namespace std;
using namespace cv;


Detector::Detector(const string& model_file,
	const string& weights_file,
	const float confidence,
	const float nms)
{
	net_ = cv::dnn::readNetFromCaffe(model_file, weights_file);
	confidence_threshold = confidence;
	nms_threshold = nms;
}



std::vector<Anchor> Detector::Detect(cv::Mat& img, cv::Size &blob_size)
{

	std::vector<AnchorGenerator> ac(_feat_stride_fpn.size());
	for (int i = 0; i < _feat_stride_fpn.size(); ++i) {
		int stride = _feat_stride_fpn[i];
		ac[i].Init(stride, anchor_cfg[stride], false);
	}


	Mat blob_input = dnn::blobFromImage(img, 1.0, blob_size, cv::Scalar(127, 127, 127), true, false, CV_32FC1);
	std::vector<Mat> targets_blobs;
	net_.setInput(blob_input, "data");


	std::vector<String>  targets_node;
	for (int i = 0; i < _feat_stride_fpn.size(); ++i)
	{
		char clsname[100]; sprintf(clsname, "face_rpn_cls_prob_reshape_stride%d", _feat_stride_fpn[i]);
		char regname[100]; sprintf(regname, "face_rpn_bbox_pred_stride%d", _feat_stride_fpn[i]);
		char ptsname[100]; sprintf(ptsname, "face_rpn_landmark_pred_stride%d", _feat_stride_fpn[i]);

		targets_node.push_back(clsname);
		targets_node.push_back(regname);
		targets_node.push_back(ptsname);
	}

	net_.forward(targets_blobs, targets_node);
	printf("forward...\n");

	std::vector<Anchor> proposals;
	int index = 0;
	for (int i = 0; i < _feat_stride_fpn.size(); ++i)
	{
		cv::Mat clsBlob = targets_blobs[index++];
		cv::Mat regBlob = targets_blobs[index++];
		cv::Mat ptsBlob = targets_blobs[index++];
		ac[i].FilterAnchor(&clsBlob, &regBlob, &ptsBlob, proposals, ratio_w, ratio_h, confidence_threshold);
	}

	// nms
	std::vector<Anchor> result;
	nms_cpu(proposals, nms_threshold, result);
	return result;
}


