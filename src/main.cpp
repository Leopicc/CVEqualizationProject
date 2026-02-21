// Author: Berno Sara

#include "utilities.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace fs = std::filesystem;   // shortcut to abbreviate std::filesystem:: into fs::


int main(int argc, char** argv) {

      // define input and output directories
      if (argc <= 2) {   // requires input_dir and output_dir, optional csv_path
            std::cerr<<"Usage: "<<argv[0]<<" <input_directory> <output_directory> [csv_path]\n";
            return 1;
      }

      std::cout << "\n--- Starting BestHistogramEqualization";
      
      const fs::path input_directory = argv[1];
      const fs::path output_directory = std::string(argv[2]) + "/output_data";
      fs::path csv_path = output_directory/"metrics.csv";
      if (argc > 3) csv_path = std::string(argv[3]) + "/metrics.csv";

      // vector of histogram equalization techniques
      const std::vector<bhe::Methods> methods = {
            bhe::Methods::GLOBAL_HE,
            bhe::Methods::CLA_HE,
            bhe::Methods::BI_HE,
            bhe::Methods::MULTI_HE,
            bhe::Methods::MATCHING_HE,
      };
      
      
      // output setup
      // for every image create a directory, inside create the original histogram and other directories for every equalization type with output image and histogram
      std::cout << "\n-- Building paths ...\n";
      
      // output root
      try {
            fs::create_directories(output_directory);
      } catch (const fs::filesystem_error& exc) {
            std::cerr << "\nOutput layout error: " << exc.what() << "\n";
            return 1;
      }

      // scan dataset for input images
      std::vector<fs::path> images;
      try {
            images = bhe::listImages(input_directory);   // recursively get all valid image paths from input_directory
      } catch (const std::exception& exc) {
            std::cerr << "\nException from reading images:" << exc.what() << "\n";
            return 1;
      }
      if (images.empty()) {
            std::cerr << "\nNo images found in " << input_directory << "\n";
            return 1;
      }

      // open CSV for metrics
      std::ofstream csv_file(csv_path);
      if (!csv_file.is_open()) {
            std::cerr << "\nCannot open CSV file: " << csv_path << "\n";
            return 1;
      }
      csv_file << "file,method,brightness,ambe,psnr,ssim,entropy,std_deviation,brisque\n";

      bhe::Params params;
      std::cout << "-- Computing equalizations ..." << std::endl;
      
      // LOOP ON IMAGES: process all images one by one
      for (const fs::path& img_path : images) {   // for every path in images, extract it as img_path and use it read only
            
            // reading the image in grayscale
            cv::Mat gray_img = cv::imread(img_path.string(), cv::IMREAD_GRAYSCALE);
            if (gray_img.empty()) {
                  std::cerr << "\nFailed to read: " << img_path << "\n";
                  continue;
            }
            
            // creation of the folder for the image
            const std::string stem = img_path.stem().string();  
            fs::path img_directory = output_directory / stem;
            try {
                  fs::create_directories(img_directory);
            } catch (const fs::filesystem_error& exc) {
                  std::cerr << "\nCannot create directory for image " << img_path << ": " << exc.what() << "\n";
                  continue;
            }
            
            // save the histogram of the original input image in the folder
            try {
                  cv::Mat original_hist = bhe::renderHistogramImage(gray_img);
                  cv::imwrite(img_directory.string()+"/"+stem+"_hist.png", original_hist);
            } catch (const std::exception& exc) {
                  std::cerr << "\nSave original error: " << img_directory << " - " << exc.what() << "\n";
                  continue;
            }


            // LOOP ON TECHNIQUES: apply every equalization method and compute metrics  
            for (bhe::Methods m : methods) {

                  // create subdirectory for the method
                  fs::path method_directory = img_directory / bhe::methodName(m);
                  try {
                        fs::create_directories(method_directory);
                  } catch (const fs::filesystem_error& exc) {
                        std::cerr << "\nCannot create directory for method " << method_directory << ": " << exc.what() << "\n";
                        continue;
                  }

                  // apply equalizations
                  cv::Mat equalized_img;
                  try {
                        //std::cout<<"\n Applying "<<bhe::methodName(m)<<" on "<<stem;
                        equalized_img = bhe::applyEqualization(gray_img, m, params);
                  } catch (const std::exception& exc) {
                        std::cerr << "\nEqualization error [" << bhe::methodName(m) << "] on " << img_path << ": " << exc.what() << "\n";
                        continue;
                  }
                  
                  // compute metrics (the reference is the original gray_img)
                  bhe::MetricsList list;
                  try {
                        list = bhe::computeMetrics(gray_img, equalized_img);   // compares the original "gray_img" and output "equalized_img"
                  } catch (const std::exception& exc) {
                        std::cerr << "\nMetric error [" << bhe::methodName(m) << "] on " << img_path << ": " << exc.what() << "\n";
                        continue;
                  }
                  
                  // save equalized output image
                  const fs::path output_img_path = method_directory / (stem + "_" + bhe::methodName(m) +".png");
                  try {
                        cv::imwrite(output_img_path.string(), equalized_img);
                  } catch (const std::exception& exc) {
                        std::cerr << "\nSave computed error: " << output_img_path << " - " << exc.what() << "\n";
                        continue;
                  }
                  
                  // save histogram of output image
                  try {
                        cv::Mat hist_img = bhe::renderHistogramImage(equalized_img);
                        const fs::path output_hist_path = method_directory / (stem + "_" + bhe::methodName(m) + "_hist" +".png");
                        cv::imwrite(output_hist_path.string(), hist_img);
                  } catch (const std::exception& exc) {
                        std::cerr << "\nHistogram error [" << bhe::methodName(m) << "] on " << output_img_path  << ": " << exc.what() << "\n";
                        continue;
                  }
                  
                  // writing on CSV
                  const std::string rel = fs::relative(img_path, input_directory).generic_string();
                  csv_file << rel << ','
                           << bhe::methodName(m) << ','
                           << list.brightness << ','
                           << list.ambe << ','
                           << list.psnr << ','
                           << list.ssim << ','
                           << list.entropy << ','
                           << list.std_deviation << ','
                           << list.brisque << '\n';
            }
      }

      // close CSV file
      csv_file.close();

      std::cout << "-- Analyzing metrics ..." << std::endl;
      // find the best equalization method for every image based on the computed metrics      
      try {
          bhe::findTheBest(csv_path, input_directory, output_directory, params);
      } catch (const std::exception& exc) {
          std::cerr << "\nMetrics comparison error: " << exc.what() << "\n";
          return 1;
      }
      
      std::cout << "\n--- Completed \n Outputs in: " << output_directory << "\n Metrics file: " << csv_path  << "\n\n";
      return 0;
}
