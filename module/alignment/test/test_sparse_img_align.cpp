#include <sparse_image_alignment.hpp>
#include <chrono>

namespace test
{

void readDepthMat(const std::string& file_name, cv::Mat& img)
{
    std::ifstream depth_reader(file_name.c_str());
    assert(depth_reader.is_open());
    img = cv::Mat(vo_kit::Cam::height(), vo_kit::Cam::width(), CV_32FC1);

    for(int y=0; y<img.rows; ++y)
    {
        float* img_p = reinterpret_cast<float*>(img.data + y*img.step[0]) ; // 注意这里的step的使用，增加的是字节数，因此不能在转化到float之后再加
        for(int x=0; x<img.cols; x++)
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

    
    std::ofstream ofs("module/alignment/Sparse_ImgAlign_estimate.txt");
    vo_kit::SparseImgAlign aligner(MaxLevel, MinLevel, 30, vo_kit::SparseImgAlign::GaussNewton, pattern, 0);
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> end;

    for(int i=0; i<num_dataset; i++)
    {
        //* read image
        std::string filename = data_path+"/img/"+sequence.image_names_[i]+"_0.png";
        cv::Mat img_gray(cv::imread(filename,0));
        assert(!img_gray.empty());

        filename = data_path+"/depth/"+sequence.image_names_[i]+"_0.depth";
        cv::Mat img_depth;
        readDepthMat(filename, img_depth);

        //* extract feature
        std::vector<cv::KeyPoint> fast_point;
        std::vector<cv::KeyPoint> good_fast_point;
        good_fast_point.resize(grid_size*grid_size);

        cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(50, true);
	    fast->detect(img_gray, fast_point);

        const int cols_per_grid = img_gray.cols/grid_size;
	    const int rows_per_grid = img_gray.rows/grid_size;
        // grid choose feature
	    for(auto iter = fast_point.begin(); iter != fast_point.end(); ++iter)
	    {
	    	int grid_x = static_cast<int>((*iter).pt.x) / cols_per_grid;
	    	int grid_y = static_cast<int>((*iter).pt.y) / rows_per_grid;
	    	int k = grid_y*grid_size + grid_x; // in k-th grid

	    	if(good_fast_point[k].response < (*iter).response)
	    		good_fast_point[k] = (*iter);
	    }
        // delete blank grid
	    for(auto iter = good_fast_point.begin(); iter!=good_fast_point.end();)
	    {
	    	if( (*iter).pt.x == 0 && (*iter).pt.y == 0)
	    	{
	    		good_fast_point.erase(iter);
	    	}
	    	else
	    	{
	    		++iter;
	    	}
	    }

#ifdef __OUTPUT_MESSAGES_
        std::cout<<"good feature size: "<< good_fast_point.size()<< std::endl;
#endif
        //* first frame
        if(i == 0)
        {
            frame_ref = vo_kit::Frame::create(img_gray, img_depth, MaxLevel);
            Sophus::SE3d T_w_gt(sequence.q_[i], sequence.t_[i]);
            frame_ref->setPose(T_w_gt.inverse()); // first as groundtruth

            frame_ref->features = good_fast_point; // set feature
            ofs <<"# \t tx \t ty \t tz \t qx \t qy \t qz \t qw"<<std::endl;
            continue;
        }

        vo_kit::Frame::Ptr frame_cur = vo_kit::Frame::create(img_gray, img_depth, MaxLevel);
        frame_cur->features = good_fast_point;
        frame_cur->setPose(frame_ref->getPose());  // last frame T as initial T

        start = std::chrono::steady_clock::now();
        size_t is_good = aligner.run(frame_ref, frame_cur); // image alignment
        end = std::chrono::steady_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count();

#ifdef _OUTPUT_MESSAGES_
        std::cout<<"NO."<<i;
        std::cout<<" Number of pixels used in alignment is "<<is_good;
        std::cout<<" Cost time is "<<duration<<std::endl;
#endif
        Sophus::SE3d T_w_cur_et = frame_cur->getPose().inverse();

        ofs << T_w_cur_et.translation().transpose() << " "
            << T_w_cur_et.unit_quaternion().vec().transpose()<<" "<<T_w_cur_et.unit_quaternion().w()<<std::endl;

        //* compute error
        Sophus::SE3d T_w_cur_gt(sequence.q_[i], sequence.t_[i]);
        // Sophus::SE3d T_cur_w_gt = T_w_cur_gt.inverse();

        Sophus::SE3d T_error = T_w_cur_et.inverse()*T_w_cur_gt;
        translation_error.push_back(T_error.translation().norm());

        frame_ref = frame_cur; // current as next reference
    }
    ofs.close();

    error = 0;
    ofs.open("module/alignment/Sparse_ImgAlign_trans_error.txt");
    for(auto it = translation_error.begin(); it != translation_error.end(); ++it)
    {    ofs << *it <<std::endl;  error += *it; }
    ofs.close();

};

} // end namespace test

int main()
{
    std::string dataset_dir(std::getenv("SVO_DATASET_DIR"));
    double error;
    test::Sparse_ImgAlign_Test(dataset_dir, error);
    return 0;
}