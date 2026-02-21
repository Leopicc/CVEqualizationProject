// Author: Carretta Francesco

#ifndef METRICS_HPP
#define METRICS_HPP

#include <opencv2/core.hpp>

namespace bhe {

    double meanIntensity(const cv::Mat& processed_img);
    
    double ambe(const cv::Mat& reference_img, const cv::Mat& processed_img);
        
    double psnr(const cv::Mat& reference_img, const cv::Mat& processed_img);
    
    double ssim(const cv::Mat& reference_img, const cv::Mat& processed_img);
    
    double entropy(const cv::Mat& processed_img);
    
    double stdDeviation(const cv::Mat& processed_img);
    
    double brisque(const cv::Mat& processed_img);
}

#endif //METRICS_HPP


