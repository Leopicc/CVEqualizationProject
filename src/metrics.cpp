// Author: Carretta Francesco

#include "metrics.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/quality.hpp>
#include <cmath>
#include <limits>


// --- Compute the mean brightness ---
double bhe::meanIntensity(const cv::Mat& processed_img) {
    
    return cv::mean(processed_img)[0];   // mean() returns the mean value of array elements; [0] because only 1 channel
}



// --- Compute the AMBE ---
double bhe::ambe(const cv::Mat& reference_img, const cv::Mat& processed_img) {

    const double mi_ref = bhe::meanIntensity(reference_img);
    const double mi_pro = bhe::meanIntensity(processed_img);
    
    return std::abs(mi_pro - mi_ref);   // difference of mean intensity between two images
}



// --- Compute the PSNR ---
double bhe::psnr(const cv::Mat& reference_img, const cv::Mat& processed_img) {
    
    const double max_val = 255.0; //CV_8U
    
    cv::Scalar s = cv::quality::QualityPSNR::compute(reference_img, processed_img, cv::noArray(), max_val);   // returns a cv::Scalar (one for every channel)
    return s[0];   // only grayscale channel
}



// --- Compute the SSIM ---
double bhe::ssim(const cv::Mat& reference_img, const cv::Mat& processed_img) {
    
    cv::Scalar s = cv::quality::QualitySSIM::compute(reference_img, processed_img, cv::noArray());
    return s[0];   // only grayscale channel
}



// --- Compute the Shannon entropy ---
double bhe::entropy(const cv::Mat& processed_img) {

    // histogram parameters
    const int channels[] = {0};
    const int hist_size[] = {256};
    const float range[] = {0.f, 256.f};
    const float* ranges[] = {range};

    // histogram computation
    cv::Mat hist;
    cv::calcHist(&processed_img, 1, channels, cv::Mat(), hist, 1, hist_size, ranges, true, false);

    // total number of pixels
    const double tot_pixel = static_cast<double>(processed_img.total());
    if (tot_pixel <= 0.0) return 0.0;

    // entropy computation
    double shannon_entr = 0.0;
    for (int i = 0; i < 256; ++i) {
        const double p = hist.at<float>(i) / tot_pixel;   // p = relative frequency of gray level i (probability p_i)
        if (p > 0.0) shannon_entr -= p * (std::log(p) / std::log(2.0));   // p * ln(p)/ln(2) = p * log2(p)
    }
    
    return shannon_entr;
}



// --- Compute the standard deviation ---
double bhe::stdDeviation(const cv::Mat& processed_img) {

    cv::Scalar mean, std_dev;
    cv::meanStdDev(processed_img, mean, std_dev);
    
    return std_dev[0];
}



// --- Compute the BRISQUE ---
double bhe::brisque(const cv::Mat& processed_img) {

    // Model/range BRISQUE (already defined in opencv quality library):
    static std::string model = "/usr/local/share/opencv4/quality/brisque_model_live.yml";
    static std::string range = "/usr/local/share/opencv4/quality/brisque_range_live.yml";

    // Cache instantiation
    static cv::Ptr<cv::quality::QualityBRISQUE> quality_b;
    if (!quality_b) quality_b = cv::quality::QualityBRISQUE::create(model, range);

    cv::Scalar sc = quality_b -> compute(processed_img);
    return sc[0];   // only grayscale channel
}
