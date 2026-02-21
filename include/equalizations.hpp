// Author: Piccoli Leonardo

#ifndef EQUALIZATIONS_HPP
#define EQUALIZATIONS_HPP

#include <opencv2/core.hpp>

namespace bhe {

    cv::Mat applyGlobalHE(const cv::Mat& gray_img);
    
    cv::Mat applyCLAHE(const cv::Mat& gray_img, double clip_limit, cv::Size tile_grid);

    cv::Mat applyBIHE(const cv::Mat& gray_img);

    cv::Mat applyMultiHE(const cv::Mat& gray_img, int multi_parts);

    cv::Mat applyMatching(const cv::Mat& gray_img, float mean, float variance);
}

#endif //EQUALIZATIONS_HPP
