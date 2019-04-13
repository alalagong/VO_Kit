#include <sparse_image_alignment.hpp>


namespace test
{

void readDepthMat(const std::string& file_name, cv::Mat& img)
{
    std::ifstream depth_reader(file_name.c_str());
    assert(depth_reader.is_open());
    img = cv::Mat(vo_kit::Cam::height(), vo_kit::Cam::width(), CV_32FC1);

    for(int y=0; y<img.rows; ++y)
    {
        float* img_p = reinterpret_cast<float*>(img.data) + y*img.step[0];
        for(int x=0; x<img.rows; x++)
        {
            float val =0 ;
            if(depth_reader.eof())
            {
                std::cerr<<"Reading data from "<<file_name<<" occurs error"<<std::endl;
                return ;
            }
            depth_reader >> val;
            img_p[x] = val/100.0; // cm 2 m
        }
    }

}



void Sparse_ImgAlign_Test(const std::string& data_path, double& error)
{
    vo_kit::DatasetReader sequence(data_path);
    int num_dataset = sequence.size();
    std::list<double> translation_error; 
    vo_kit::Frame::Ptr frame_ref; // reference frame, update every iteration

    // setting
    int MaxLevel = 4;
    int MinLevel = 2;
    const int grid_size = 5;
    std::vector<std::pair<int, int> > pattern;
    for(int i=0; i<4; ++i)
        for(int j=0; j<4; ++j)
            pattern.push_back(std::make_pair(i-2, j-2));  // svo
    //* DSO pattern
    // {
        // pattern.push_back(std::make_pair(-2, 0));
        // pattern.push_back(std::make_pair(-1,-1));
        // pattern.push_back(std::make_pair(-1, 1));
        // pattern.push_back(std::make_pair( 0,-2));
        // pattern.push_back(std::make_pair( 0, 0));
        // pattern.push_back(std::make_pair( 0, 2));
        // pattern.push_back(std::make_pair( 1,-1));
        // pattern.push_back(std::make_pair( 2, 0));
    // }

    
    std::ofstream ofs("Sparse_ImgAlign_estimate.txt");
    vo_kit::SparseImgAlign aligner(MaxLevel, MinLevel, 30, vo_kit::SparseImgAlign::GaussNewton, pattern, 0);

    for(int i=0; i<num_dataset; i++)
    {
        // read image
        std::string filename = data_path+"/img/"+sequence.image_names_[i]+"_0.png";
        cv::Mat img_gray(cv::imread(filename,0));
        assert(!img_gray.empty());

        filename = data_path+"/depth/"+sequence.image_names_[i]+"_0.depth";
        cv::Mat img_depth;
        readDepthMat(filename, img_depth);

        // extract feature
        std::vector<cv::KeyPoint> fast_point;
        std::vector<cv::KeyPoint> good_fast_point;
        good_fast_point.resize(grid_size*grid_size);
        
        

        if(i == 0)
        {
            frame_ref = vo_kit::Frame::create(img_gray, img_depth, MaxLevel);
            Sophus::SE3d T_w_gt(sequence.q_[i], sequence.t_[i]);
            frame_ref->setPose(T_w_gt.inverse()); // first as groundtruth



        }

    }


};

} // end namespace test

int main()
{

}