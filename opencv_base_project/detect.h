#include "anchor_generator.h"

class Detector {
 public:
  Detector(const std::string& model_file,
           const std::string& weights_file,          
		   const float confidence,
       const float nms);

  std::vector<Anchor> Detect(cv::Mat& img, cv::Size &blob_size);
 private:
  cv::dnn::Net net_;
  int num_channels_;
  float confidence_threshold;
  float nms_threshold;

  float ratio_w=1.0;
  float ratio_h=1.0;
 
};