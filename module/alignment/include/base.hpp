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

    typedef std::shared_ptr<Frame> Ptr;
    
    size_t                      num_max_level_;     //!< Max numbers of levels of pyramid
    std::vector<cv::KeyPoint>   features;           //!< the corner of image
    static int                  frame_count_;       //!< Count the number of frames


    void createImgPyr(const cv::Mat& img);
    void setPose(const Sophus::SE3d& pose);
    Sophus::SE3d getPose() const;
    cv::Mat getImage(size_t level) const; 
    float getDepth(int u, int v) const;

    inline static Ptr create(const cv::Mat& img, const cv::Mat& depth, const double time_stamp)
    {
        return Ptr(new Frame(img, depth, time_stamp));
    }
    

protected:

    Frame(const cv::Mat& img, const cv::Mat& depth, const double time_stamp);
    Frame(const Frame&) = delete;
    Frame &operator=(const Frame&) = delete;

    Sophus::SE3d     T_c_w_;             //!< Pose Transform from world to camera
    ImgPyr          img_pyr_;            //!< Image pyramid
    cv::Mat         img_depth_;          //!< image of depth
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
        while(!data_stream.eof())
        {
            string s;
            std::getline(data_stream, s);
            if(!s.empty()){
                std::stringstream ss;
                ss<<s;
                double times, tx, ty, tz, qx, qy, qz, qw;
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
                q_.push_back(Eigen::Quaterniond(qx, qy, qz, qw));
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
            std::cerr << " Index(" << index << ") is out of scape, max should be (0~" << N - 1 << ")";
            return false;
        }
        timestamp = timestamps_[index];
        filename = image_names_[index];
        t = t_[index];
        q = q_[index];
        return true;
    }



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