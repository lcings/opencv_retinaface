#include"detect.h"

using namespace std;

int main(int argc, char** argv) {

	const string proto = "mnet.prototxt";
	const string model = "mnet.caffemodel";
	const float confidence = 0.6;
	const float nms_threshold = 0.4;
	Detector detector(proto, model, confidence, nms_threshold);
	cv::Mat img = cv::imread("demo.jpg");

	std::vector<Anchor> result = detector.Detect(img, img.size());
	printf("face num:%d\n", result.size());
	for (int i = 0; i < result.size(); i++) {
		cv::rectangle(img, cv::Point((int)result[i].finalbox.x, (int)result[i].finalbox.y), cv::Point((int)result[i].finalbox.width, (int)result[i].finalbox.height), cv::Scalar(0, 255, 255), 2, 8, 0);
		cv::circle(img, result[i].pts[0], 2, cv::Scalar(0, 255, 0), -1);
		cv::circle(img, result[i].pts[1], 2, cv::Scalar(0, 255, 0), -1);
		cv::circle(img, result[i].pts[2], 2, cv::Scalar(0, 255, 0), -1);
		cv::circle(img, result[i].pts[3], 2, cv::Scalar(0, 255, 0), -1);
		cv::circle(img, result[i].pts[4], 2, cv::Scalar(0, 255, 0), -1);
	}

	cv::imshow("show", img);
	cv::waitKey(0);
	return 0;
}
