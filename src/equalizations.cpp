// Authors: Piccoli Leonardo

#include "equalizations.hpp"
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>


// --- Global Histogram Equalization ---
cv::Mat bhe::applyGlobalHE(const cv::Mat& gray_img) {

    cv::Mat output_img;
    cv::equalizeHist(gray_img, output_img);
    
    return output_img;
}



// --- Contrast Limited Adaptive Histogram Equalization ---
cv::Mat bhe::applyCLAHE(const cv::Mat& gray_img, double clip_limit, cv::Size tile_grid) {

    cv::Mat output_img;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	
    clahe->setClipLimit(clip_limit);
    clahe->setTilesGridSize(tile_grid);
    clahe->apply(gray_img, output_img);
    
    return output_img;
}



// ----- HELPERS -----

static void calcHistCDF(const cv::Mat& gray_img, std::array<double, 256>& cumulative_dis_func) {   // use double array compatible with function to calculate hist OpenCV
	
	static const int   channels[] = {0};
    static const int   histSize[] = {256};
    static const float range[]    = {0.f, 256.f};
    static const float* ranges[]  = {range};
	
	cv::Mat1f histMat;   // final type where we'll record the histogram of the img
	cv::calcHist(&gray_img, 1, channels, cv::Mat(), histMat, 1, histSize, ranges, true, false);
	
	double sum = 0.0;
    for (int i = 0; i < 256; ++i) {
        sum += histMat(i);
        cumulative_dis_func[i] = sum;
	}
	
	// normalization
	for (int i = 0; i < 256; ++i)
	    cumulative_dis_func[i] /= sum;
}


// For Multi and BPBi HE
static std::vector<int> cutsFromCDF(const std::array<double, 256>& cdf, double total_pixels, int n_cuts) {

	std::vector<int> where_to_cuts;
	where_to_cuts.push_back(0);   // first cut at position pixel 0
	
	for (int k = 1; k < n_cuts; k = k+1) {
		double target_intensity = (double)k*total_pixels / (double)n_cuts;   // find the threshold for our cuts (25%, 50%... of pixels)
		int i=0;
		while (i < 256 && cdf[i] < target_intensity)
		    i=i+1;
		
		if (i <= where_to_cuts.back())
			i = std::min(255, where_to_cuts.back() + 1);
		where_to_cuts.push_back(i);
	}
	
	where_to_cuts.push_back(255);
	
	return where_to_cuts;
}


static cv::Mat buildLookUpTable(const std::array<double, 256>& cumulative_dis_func, const std::vector<int>& where_to_cuts) {

	cv::Mat lut(1, 256, CV_8U);	  // create a 1x256 matrix, every element is a uchar
	uchar* lut_pointer = lut.ptr<uchar>();   // a pointer to the first element of the LUT. We can access every element of our LUT starting from this pointer
	
	// default values for our LUT
	for (int v=0; v<256; v=v+1)
		lut_pointer[v] = static_cast<uchar>(v);
	
	// define the segments [a...b] starting from the previously find threshold
	int n_segments = static_cast<int>(where_to_cuts.size())-1;
	for (int j = 0; j < n_segments; j = j+1) {
		int a = where_to_cuts[j];
		int b = where_to_cuts[j+1];
		
		int start;
		if (j == 0)
			start = a;   // first segment includes left border
		else
			start = a + 1;    // no overlap in the following segments (they start from a+1)

		if (start > b)
			continue;   // empty segment
	
		// calculate the width of our cut (at least 1)
		int width = std::max(1, b-a);
		
		// calculate the CDF before "a" (if a==0, then put CDF = 0)
		double cdf_before;
		if (a==0)
			cdf_before = 0.0;
		else
			cdf_before = cumulative_dis_func[a-1];
			
		// count the number of pixel for every cut [a...b]
		double seg_count = cumulative_dis_func[b] - cdf_before;
		if (seg_count <= 0.0)
			continue; //no pixel for a certain cut --> leave the default value
	
		// build a map for every grey level belonging to a certain cut
		for (int v = a; v <= b; v = v+1) {
		
			// find the pixels in the segment with level <=b (that is, local CDF)
			const double cdf_local = cumulative_dis_func[v] - cdf_before;
		
			// normalize local CDF [0...1] to identify the relevance inside a segment 
			const double p = cdf_local / seg_count;
		
			// remap v in [a...b] streching or compressing the local CDF
			int mapped = a + static_cast<int>(std::round(p * width));
		
			lut_pointer[v] = static_cast<uchar>(mapped);
		}
	}
	
	return lut;
}


// From matching (other distributions are possible such as exponential, logarithmic, ...)
static std::vector<float> generateInverseGaussianPDF(float mean, float variance, int levels = 256) {   // inverse dicrete gaussian distribution on 256 levels

    std::vector<float> probability_dis_func(levels);
    float sigma = std::sqrt(variance);
    float sum = 0.0f;
    
    // PDF
    for (int i = 0; i < levels; i++) {
        float x = static_cast<float>(i);
        probability_dis_func[i] = 1-std::exp(-0.5f * std::pow((x - mean) / sigma, 2.0f)) / (sigma * std::sqrt(2.0f * CV_PI));   //
        sum += probability_dis_func[i];
    }
    
    // normalization
    for (int i = 0; i < levels; i++)
        probability_dis_func[i] /= sum;

    // CDF
    std::vector<float> cumulative_dis_func(levels, 0.0f);
    sum = 0;
    for (int i = 0; i < levels; i++) {
        sum += probability_dis_func[i];
        cumulative_dis_func[i] = sum;
    }
    
    return cumulative_dis_func;
}

// ----- END OF HELPERS -----



// --- Brigthness Preserving Bi-histogram Equalization (BiHE) ---
cv::Mat bhe::applyBIHE(const cv::Mat& gray_img) {

	// T threshold = average luminance
    int T = static_cast<int>(std::lround(cv::mean(gray_img)[0]));
    T = std::clamp(T, 0, 254);

	// normalized CDF
	std::array<double, 256> cdf;
	calcHistCDF(gray_img, cdf);

	// cuts for BIHE = {0, T, 255}
	std::vector<int> cuts = {0, T, 255};

	// calculate LUT and return new img
	cv::Mat lut = buildLookUpTable(cdf, cuts);
	cv::Mat output_img;
	cv::LUT(gray_img, lut, output_img);
	
	return output_img; 
}



// --- Multi Histogram Equalization ---
cv::Mat bhe::applyMultiHE(const cv::Mat& gray_img, int multi_parts) {

	const int n_cuts = std::max(2, multi_parts);   //read in how many segment to cut
	std::array<double,256> original_cdf;   // original CDF
    calcHistCDF(gray_img, original_cdf);
	
	const double total = original_cdf[255];
    if (total <= 0.0) return gray_img.clone();   // if there is something wrong and the number of total pixel il = to zero, exit, returning the initial img
	
	std::vector<int> cuts = cutsFromCDF(original_cdf, total, n_cuts);
	cv::Mat lut = buildLookUpTable(original_cdf, cuts);
	
	cv::Mat output_img;
	cv::LUT(gray_img, lut, output_img);
	
	return output_img; 
}



// --- Histogram Matching/Specification ---
cv::Mat bhe::applyMatching(const cv::Mat& gray_img, float mean, float variance) {

    // original CDF
    std::array<double,256> original_cdf;
    calcHistCDF(gray_img, original_cdf);

    // gaussian CDF (other options can be available: exponantial, logarithmic, ...)
    std::vector<float> target_cdf = generateInverseGaussianPDF(mean, variance, 256);

    // lookup table
    std::vector<uchar> lut(256);
    for (int r = 0; r < 256; r++) {
        float s = original_cdf[r];
        std::vector<float>::iterator min_index = std::lower_bound(target_cdf.begin(), target_cdf.end(), s);
        int current = static_cast<int>(min_index - target_cdf.begin());

        if (current == 0)
            lut[r] = 0;
        else if (current >= 256)
            lut[r] = 255;
            
        else {
            float up = target_cdf[current];
            float dn = target_cdf[current - 1];
            if (std::fabs(up - s) < std::fabs(s - dn))
                lut[r] = current;
            else
                lut[r] = current - 1;
        }
    }

    cv::Mat output_img;
    cv::LUT(gray_img, lut, output_img);
    
    return output_img;
}
