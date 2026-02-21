// Author: Piccoli Leonardo

#ifndef PARAMS_HPP
#define PARAMS_HPP

#include <opencv2/core.hpp>

namespace bhe {

    // parameter
    struct Params {
    
        // for CLAHE
        double clip_limit  = 2.0;
        cv::Size tile_grid = {8, 8};

        // for MultiHE
        int multi_parts = 3;
        
        // for Matching
        float mean = 127.f;
        float variance = 1.f;
    };

}
#endif //PARAMS_HPP
