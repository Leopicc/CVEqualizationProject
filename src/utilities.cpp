// Author: Berno Sara

#include "utilities.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <iostream>
#include <algorithm>
#include <set>
#include <filesystem>
#include <map>
#include <tuple>
#include <limits>
#include <cmath>
#include <fstream>
#include <sstream>
#include "equalizations.hpp"
#include "metrics.hpp"

namespace fs = std::filesystem;   // shortcut to abbreviate std::filesystem:: into fs::

// --- Return the method's name as string ---
std::string bhe::methodName(bhe::Methods method) {

    switch (method) {
        case bhe::Methods::GLOBAL_HE: 	return "GLOBAL_HE";
        case bhe::Methods::CLA_HE: 		return "CLA_HE";
        case bhe::Methods::BI_HE: 		return "BI_HE";
        case bhe::Methods::MULTI_HE: 	return "MULTI_HE";
        case bhe::Methods::MATCHING_HE: return "MATCHING_HE";
    }
    
	return "UNKNOWN";
}


// --- Executes the different types of equalization techniques on an image ---
cv::Mat bhe::applyEqualization(const cv::Mat& gray_img, bhe::Methods method, const bhe::Params& params) {

    try {   // check if the image is ok
        CV_Assert(gray_img.type() == CV_8UC1);
    } catch (const cv::Exception& exc) {
        std::cerr<<"\nException in computing equalization: "<<exc.what()<<"\n";
    }

    switch (method) {
        case bhe::Methods::GLOBAL_HE: 	return bhe::applyGlobalHE(gray_img);
        case bhe::Methods::CLA_HE: 		return bhe::applyCLAHE(gray_img, params.clip_limit, params.tile_grid);
        case bhe::Methods::BI_HE: 		return bhe::applyBIHE(gray_img);
        case bhe::Methods::MULTI_HE: 	return bhe::applyMultiHE(gray_img, params.multi_parts);
        case bhe::Methods::MATCHING_HE: return bhe::applyMatching(gray_img, params.mean, params.variance);
        default: throw std::invalid_argument("Unknown equalization method.");
    }
}


// --- Computes and save all the metrics for an image ---
bhe::MetricsList bhe::computeMetrics(const cv::Mat& original_img, const cv::Mat& processed_img) {

    try {   // check if the images ok
        CV_Assert(!original_img.empty() && !processed_img.empty());
        CV_Assert(original_img.type() == CV_8UC1 && processed_img.type() == CV_8UC1);
        CV_Assert(original_img.size() == processed_img.size());
    } catch (const cv::Exception& exc) {
        std::cerr<<"\nException in computing metrics: "<<exc.what()<<"\n";
    }
    
    bhe::MetricsList list;
    list.ambe = 		 bhe::ambe(original_img, processed_img);
    list.brightness = 	 bhe::meanIntensity(processed_img);
    list.psnr = 		 bhe::psnr(original_img, processed_img);
    list.ssim = 		 bhe::ssim(original_img, processed_img);
    list.entropy = 		 bhe::entropy(processed_img);
    list.std_deviation = bhe::stdDeviation(processed_img);
    list.brisque = 		 bhe::brisque(processed_img);
    
    return list;
}


// --- Scan images ---
std::vector<std::filesystem::path> bhe::listImages(const std::filesystem::path& input_directory) {

    std::vector<fs::path> paths_list;
    static const std::set<std::string> valid_exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"};   // valid image extensions

    // push gen_path into the list only if it is a regular file with a valid image extension
    auto push_if_img = [&](const fs::path& gen_path) {
        if (!fs::is_regular_file(gen_path)) return;
        std::string ext = gen_path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (valid_exts.count(ext)) paths_list.push_back(gen_path);
    };

    if (!fs::exists(input_directory)) return paths_list;   // if "input_directory" doesn't exist return an empty vector

    for (fs::directory_entry element : fs::recursive_directory_iterator(input_directory))
            push_if_img(element.path());
		
	// sort the paths
    std::sort(paths_list.begin(), paths_list.end());
    return paths_list;
}


// --- Render histogram into an image ---
cv::Mat bhe::renderHistogramImage(const cv::Mat& gray_img, int bins, cv::Size canvas) {

    try {   // check if the image is ok
        CV_Assert(!gray_img.empty() && gray_img.type() == CV_8UC1);
        CV_Assert(bins > 0); 
    } catch (const cv::Exception& exc) {
        std::cerr<<"\nException in rendering histogram: "<<exc.what()<<"\n";
    }                                            

    // histogram parameters
    const int channels[] = {0};
    const int histSize[] = {bins};
    const float range[] = {0.f, 256.f};
    const float* ranges[] = {range};

    // compute histogram
    cv::Mat1f hist;
    cv::calcHist(&gray_img, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    // normalize values to fit in the canvas height
    double max_val = 0.0;
    cv::minMaxLoc(hist, nullptr, &max_val);   // find the highest bin
    if (max_val <= 0.0) max_val = 1.0;

    // creation of a white canvas
    cv::Mat canvas_img(canvas, CV_8UC3, cv::Scalar(255, 255, 255));
    const int width = canvas.width;
    const int height = canvas.height;
	const int bin_wid = std::max(1, width / bins);

	// draw bars
	for (int i = 0; i < bins; ++i) {
	    int x0 = i * bin_wid;
	    int x1 = std::min(width - 1, (i + 1) * bin_wid - 1);
	    int bar_height = static_cast<int>((hist(i) / max_val) * (height - 1));
	    cv::rectangle(canvas_img, cv::Point(x0, height - 1), cv::Point(x1, height - 1 - bar_height), cv::Scalar(0, 0, 0), cv::FILLED);
	}

    return canvas_img;
}


// --- Find the method that gives the best result for every image ---
void bhe::findTheBest(const std::filesystem::path& csv_path, const std::filesystem::path& input_dir, const std::filesystem::path& output_root, const bhe::Params& params) {

    // weights for the weighted fusion (in a decreasing order of importance)
    const double W_SSIM = 0.30; 
    const double W_BRIS = 0.25; 
    const double W_PSNR = 0.20; 
    const double W_AMBE = 0.15; 
    const double W_ENTR = 0.10;  

    // open CSV and skip header
    std::ifstream input_file(csv_path);
    if (!input_file.is_open()) {
        std::cerr << "\nCannot open CSV: " << csv_path << "\n";
        return;
    }
    std::string header;
    std::getline(input_file, header); // skip one header line

    // row container of the CSV fields
    struct Row {
        std::string file;
        std::string method;
        double brightness, ambe, psnr, ssim, entropy, stddev, brisque;
        Row(): brightness(0.0), ambe(0.0), psnr(0.0), ssim(0.0), entropy(0.0), stddev(0.0), brisque(0.0) {}
    };

    // read and group rows by input file (image)
    std::map<std::string, std::vector<Row> > rows_by_file;
    std::string line;

    while (std::getline(input_file, line)) {
        if (line.empty()) continue;

        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::stringstream s(line);
        Row r;
        std::string brightness_str, ambe_str, psnr_str, ssim_str, entropy_str, stddev_str, brisque_str;

        // extract CSV fields
        std::getline(s, r.file,   ',');
        std::getline(s, r.method, ',');
        std::getline(s, brightness_str, ',');
        std::getline(s, ambe_str, ',');
        std::getline(s, psnr_str, ',');
        std::getline(s, ssim_str, ',');
        std::getline(s, entropy_str, ',');
        std::getline(s, stddev_str, ',');
        std::getline(s, brisque_str, ',');

        // convert numeric fields, if conversion fails, the value remains 0.0
        try { r.brightness = std::stod(brightness_str); } catch (...) {}
        try { r.ambe = std::stod(ambe_str); } catch (...) {}
        try { r.psnr = std::stod(psnr_str); } catch (...) {}
        try { r.ssim = std::stod(ssim_str); } catch (...) {}
        try { r.entropy = std::stod(entropy_str); } catch (...) {}
        try { r.stddev = std::stod(stddev_str); } catch (...) {}
        try { r.brisque = std::stod(brisque_str); } catch (...) {}

        if (!r.file.empty()) {
            rows_by_file[r.file].push_back(r);
        }
    }
    input_file.close();

    // normalize metrics per-image and select the winner by weighted score
    std::map<std::string, Row> best; // stores, per file, the winning row/method

    for (std::map<std::string, std::vector<Row> >::iterator it = rows_by_file.begin();
         it != rows_by_file.end(); ++it) {

        const std::string& fname = (*it).first;
        std::vector<Row>& vec    = (*it).second;
        if (vec.empty()) continue;

        // 1) Compute per-image min/max for each metric among its methods
        double miS = +std::numeric_limits<double>::infinity(), maS = -std::numeric_limits<double>::infinity(); // SSIM
        double miP = +std::numeric_limits<double>::infinity(), maP = -std::numeric_limits<double>::infinity(); // PSNR
        double miE = +std::numeric_limits<double>::infinity(), maE = -std::numeric_limits<double>::infinity(); // Entropy
        double miB = +std::numeric_limits<double>::infinity(), maB = -std::numeric_limits<double>::infinity(); // BRISQUE
        double miA = +std::numeric_limits<double>::infinity(), maA = -std::numeric_limits<double>::infinity(); // AMBE

        for (size_t i = 0; i < vec.size(); ++i) {
            const Row& r = vec[i];
            if (std::isfinite(r.ssim))    { if (r.ssim    < miS) miS = r.ssim;    if (r.ssim    > maS) maS = r.ssim;    }
            if (std::isfinite(r.psnr))    { if (r.psnr    < miP) miP = r.psnr;    if (r.psnr    > maP) maP = r.psnr;    }
            if (std::isfinite(r.entropy)) { if (r.entropy < miE) miE = r.entropy; if (r.entropy > maE) maE = r.entropy; }
            if (std::isfinite(r.brisque)) { if (r.brisque < miB) miB = r.brisque; if (r.brisque > maB) maB = r.brisque; }
            if (std::isfinite(r.ambe))    { if (r.ambe    < miA) miA = r.ambe;    if (r.ambe    > maA) maA = r.ambe;    }
        }

        // if a metric has no finite values for this image, force min=max=0 so that we later assign a neutral value (0.5)
        if (!std::isfinite(miS) || !std::isfinite(maS)) { miS = 0.0; maS = 0.0; }
        if (!std::isfinite(miP) || !std::isfinite(maP)) { miP = 0.0; maP = 0.0; }
        if (!std::isfinite(miE) || !std::isfinite(maE)) { miE = 0.0; maE = 0.0; }
        if (!std::isfinite(miB) || !std::isfinite(maB)) { miB = 0.0; maB = 0.0; }
        if (!std::isfinite(miA) || !std::isfinite(maA)) { miA = 0.0; maA = 0.0; }

        // 2) Compute the weighted score for each method and keep the best (highest score)
        size_t best_idx = 0;
        double best_score = -1.0;

        // initialize with the first method
        {
            const Row& r0 = vec[0];
            double ssimN, brisN, psnrN, ambeN, entrN;

            // SSIM (maximize)
            if (std::isfinite(r0.ssim) && (maS > miS)) ssimN = (r0.ssim - miS) / (maS - miS);
            else if (maS > miS) ssimN = 0.5;   // value invalid but range exists -> neutral
            else ssimN = 0.0;   // no range -> neutral/penalize

            // BRISQUE (minimize)
            if (std::isfinite(r0.brisque) && (maB > miB)) brisN = (maB - r0.brisque) / (maB - miB);
            else if (maB > miB) brisN = 0.5;
            else brisN = 0.0;

            // PSNR (maximize)
            if (std::isfinite(r0.psnr) && (maP > miP)) psnrN = (r0.psnr - miP) / (maP - miP);
            else if (maP > miP) psnrN = 0.5;
            else psnrN = 0.0;

            // AMBE (minimize)
            if (std::isfinite(r0.ambe) && (maA > miA)) ambeN = (maA - r0.ambe) / (maA - miA);
            else if (maA > miA) ambeN = 0.5;
            else ambeN = 0.0;

            // Entropy (maximize)
            if (std::isfinite(r0.entropy) && (maE > miE)) entrN = (r0.entropy - miE) / (maE - miE);
            else if (maE > miE) entrN = 0.5;
            else entrN = 0.0;

            best_score = W_SSIM*ssimN + W_BRIS*brisN + W_PSNR*psnrN + W_AMBE*ambeN + W_ENTR*entrN;
        }

        // compare with the remaining methods
        for (size_t i = 1; i < vec.size(); ++i) {
            const Row& r = vec[i];

            double ssimN, brisN, psnrN, ambeN, entrN;

            // SSIM
            if (std::isfinite(r.ssim) && (maS > miS)) ssimN = (r.ssim - miS) / (maS - miS);
            else if (maS > miS) ssimN = 0.5;
            else ssimN = 0.0;

            // BRISQUE
            if (std::isfinite(r.brisque) && (maB > miB)) brisN = (maB - r.brisque) / (maB - miB);
            else if (maB > miB) brisN = 0.5;
            else brisN = 0.0;

            // PSNR
            if (std::isfinite(r.psnr) && (maP > miP)) psnrN = (r.psnr - miP) / (maP - miP);
            else if (maP > miP) psnrN = 0.5;
            else psnrN = 0.0;

            // AMBE
            if (std::isfinite(r.ambe) && (maA > miA)) ambeN = (maA - r.ambe) / (maA - miA);
            else if (maA > miA) ambeN = 0.5;
            else ambeN = 0.0;

            // entropy
            if (std::isfinite(r.entropy) && (maE > miE)) entrN = (r.entropy - miE) / (maE - miE);
            else if (maE > miE) entrN = 0.5;
            else entrN = 0.0;

            double score = W_SSIM*ssimN + W_BRIS*brisN + W_PSNR*psnrN + W_AMBE*ambeN + W_ENTR*entrN;

            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        // keep the best row for this image
        best[fname] = vec[best_idx];
    }

    // re-apply the winning method and save outputs into BEST/ directory
    std::filesystem::path best_dir = output_root / "BEST";
    try {
        std::filesystem::create_directories(best_dir);
    } catch (const std::filesystem::filesystem_error& exc) {
        std::cerr << "\nOutput directory error: " << exc.what() << "\n";
        return;
    }

    for (std::map<std::string, Row>::const_iterator it = best.begin(); it != best.end(); ++it) {
        const std::string& f_name = (*it).first;
        const Row& best_row = (*it).second;

        // read original image (grayscale)
        cv::Mat gray = cv::imread((input_dir / best_row.file).string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty()) continue;

        // decode method enum from string
        bhe::Methods m;
        if (best_row.method == "GLOBAL_HE") m = bhe::Methods::GLOBAL_HE;
        else if (best_row.method == "CLA_HE") m = bhe::Methods::CLA_HE;
        else if (best_row.method == "BI_HE") m = bhe::Methods::BI_HE;
        else if (best_row.method == "MULTI_HE") m = bhe::Methods::MULTI_HE;
        else if (best_row.method == "MATCHING_HE") m = bhe::Methods::MATCHING_HE;
        else continue; // unknown method label

        // re-apply the winning equalization
        cv::Mat best_img = bhe::applyEqualization(gray, m, params);

        // save best image and its histogram
        std::filesystem::path out_img  = best_dir / (std::filesystem::path(f_name).stem().string() + "_" + best_row.method + ".png");
        std::filesystem::path out_hist = best_dir / (std::filesystem::path(f_name).stem().string() + "_" + best_row.method + "_hist.png");
        cv::imwrite(out_img.string(),  best_img);
        cv::imwrite(out_hist.string(), bhe::renderHistogramImage(best_img));
    }
}
