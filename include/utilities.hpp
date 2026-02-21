// Author: Berno Sara

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <filesystem>
#include "params.hpp"
#include "methods.hpp"
#include "metricsList.hpp"

namespace bhe {

    std::string methodName(bhe::Methods method);
    
    cv::Mat applyEqualization(const cv::Mat& gray_img, bhe::Methods method, const bhe::Params& params);

    bhe::MetricsList computeMetrics(const cv::Mat& original_img, const cv::Mat& processed_img);
        
    std::vector<std::filesystem::path> listImages(const std::filesystem::path& input_dir);

    cv::Mat renderHistogramImage(const cv::Mat& gray_img, int bins = 256, cv::Size canvas = {512, 300});
    
    void findTheBest(const std::filesystem::path& csv_path, const std::filesystem::path& input_dir, const std::filesystem::path& output_root, const bhe::Params& params);
}

#endif //UTILITIES_HPP
