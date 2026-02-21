// Author: Carretta Francesco

#ifndef METRICSLIST_HPP
#define METRICSLIST_HPP

namespace bhe {

    // computed metrics
    struct MetricsList {
    
        // Mean intensity of an image
        double brightness= 0.0;
        
        // Absolute Mean Brightness Error (original - processed)
        double ambe = 0.0;        
        
        // Peak Signal to Noise Ratio (original - processed)
        double psnr = 0.0;
        
        // Structural SIMilarity index (original - processed)
        double ssim = 0.0;
        
        // Shannon entropy
        double entropy = 0.0;
        
        // Standard deviation
        double std_deviation = 0.0;
        
        // Blind/Referenceless Image Spatial Quality Evaluator
        double brisque = 0.0;
    };
    
}
#endif //METRICSLIST_HPP
