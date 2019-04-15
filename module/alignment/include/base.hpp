#ifndef _BASE_HPP_
#define _BASE_HPP_ 

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <eigen3/Eigen/Core>

#include <vector>
#include <memory>
#include <string>

#include <math_base.hpp>

namespace vo_kit
{

typedef std::vector<cv::Mat> ImgPyr;

class Cam{

public:
    // only one
    static Cam& getCamera()
    {
        static Cam instance;
        return instance;
    }
    static double& fx() { return getCamera().fx_; }
    static double& fy() { return getCamera().fy_; }
    static double& cx() { return getCamera().cx_; }
    static double& cy() { return getCamera().cy_; }
    static int& height() { return getCamera().height_; }
    static int& width()  { return getCamera().width_;  }
    static Eigen::Matrix3d& K() { return getCamera().K_; }

    static Eigen::Vector3d pixel2unitPlane(const double& u, const double& v)
    {
        Eigen::Vector3d xyz;
        xyz[0] = (u - cx())/fx();
        xyz[1] = (v - cy())/fy();
        xyz[2] = 1.0;
        return xyz;
    }

    static Eigen::Vector2d project(const Eigen::Vector3d& p)
    {
        return (K()*(p/p[2])).head(2);
    }
private:
    //TODO: read from files
    Cam():
    fx_(315.5),
    fy_(315.5),
    cx_(376.0),
    cy_(240.0),
    height_(480),
    width_(752)
    {
        K_ <<   fx_,    0,      cx_,
                0,      fy_,     cy_,
                0,      0,      1; 
    }
    
    Cam( Cam const&);
    void operator=(Cam const&);
        
    double fx_, fy_, cx_, cy_;
    int height_, width_;
    Eigen::Matrix3d K_;
};



class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Frame> Ptr;
    
    size_t                      num_max_level_;     //!< Max numbers of levels of pyramid
    std::vector<cv::KeyPoint>   features;           //!< the corner of image
    static int                  frame_count_;       //!< Count the number of frames


    void createImgPyr(const cv::Mat& img);
    void setPose(const Sophus::SE3d& pose);
    inline Sophus::SE3d getPose() const { return T_c_w_; }
    inline const cv::Mat& getImage(size_t level) const { assert(level <= num_max_level_ && level > 0);  return img_pyr_[level]; }  
    inline float getDepth(int u, int v) const{ return img_depth_.at<float>(v, u);}

    inline static Ptr create(const cv::Mat& img, const cv::Mat& depth, const size_t max_pyr)
    {
        return Ptr(new Frame(img, depth, max_pyr));
    }
    

protected:

    Frame(const cv::Mat& img, const cv::Mat& depth, const size_t max_pyr);
    Frame(const Frame&) = delete;
    Frame &operator=(const Frame&) = delete;

    Sophus::SE3d     T_c_w_;             //!< Pose Transform from world to camera
    cv::Mat         img_depth_;          //!< image of depth
    ImgPyr          img_pyr_;            //!< Image pyramid

};



class DatasetReader
{
public:
    DatasetReader(const std::string& path):dataset_file(path)
    {
        size_t found = path.find_last_of("/\\");
        if(found+1 != path.size())
            dataset_file += dataset_file.substr(found, 1);

        dataset_file += "trajectory.txt";
        std::ifstream  data_stream;
        data_stream.open(dataset_file.c_str());
        assert(data_stream.is_open());

        while(!data_stream.eof())
        {
            std::string s;
            std::getline(data_stream, s);
            if(!s.empty()){
                std::stringstream ss;
                ss<<s;
                double times, tx, ty, tz, qx, qy, qz, qw;
                Eigen::Quaterniond q;
                std::string file_name;
                ss >> times;
                timestamps_.push_back(times);
                ss >> file_name;
                image_names_.push_back(file_name);
                ss >> tx;
                ss >> ty;
                ss >> tz;
                t_.push_back(Eigen::Vector3d(tx, ty, tz));
                ss >> qx;
                ss >> qy;
                ss >> qz;
                ss >> qw;
                q = Eigen::Quaterniond(qw, qx, qy, qz);
                q.normalize();
                q_.push_back(q);
            }
        }
        
        data_stream.close();

        N_ = timestamps_.size();
        if(N_ == 0)
            std::cout<<"Nothing in datasets "<< dataset_file <<std::endl;
        else
            std::cout<<"Find items in datasets: num = "<<N_<<std::endl;
        
    }

    bool readByIndex(size_t index, double& timestamp, std::string& filename, Eigen::Vector3d& t, Eigen::Quaterniond& q)
    {
        if(index >= N_)
        {
            std::cerr << " Index(" << index << ") is out of scape, max should be (0~" << N_ - 1 << ")";
            return false;
        }
        timestamp = timestamps_[index];
        filename = image_names_[index];
        t = t_[index];
        q = q_[index];
        return true;
    }

    int size() {return N_;}

public:

    size_t N_;
    std::string dataset_file;
    std::vector<double> timestamps_;
    std::vector<std::string> image_names_;  // include rgb and depth
    std::vector<Eigen::Vector3d> t_;
    std::vector<Eigen::Quaterniond> q_;
    
};

} // end namespace vo_kit

#endif //_BASE_HPP_