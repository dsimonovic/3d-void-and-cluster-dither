// Generating 3D dithering pattern (blue noise) following the paper of R. Ulichney (1993)
//
// The 2D algorithm presented in the paper is verbatim generalized to 3D.
// 
// R. Ulichney, “The Void-and-Cluster Method for Generating Dither Arrays”,
// Human Vision, Visual Processing, and Digital Display IV, J. Allebach and B. Rogowitz, eds., Proc. SPIE 1913, pp. 332-343, 1993.
// Author's version of the paper can be found at: http://cv.ulichney.com/papers/1993-void-cluster.pdf
// 
// INPUT:  Modifiable parameters are at the top of the cpp file
// 
// OUTPUT: 3D pixel matrix is saved as layers of images.
// 
// DEPENDENCY: OpenCV and STD. Tested with C++17  
// 
// Caveat. Phase 3 is mathematically no different than Phase 2, thus, those 2 are merged together
// (Finding largest cluster center of zeros, is equivalent to finding largest void center of ones)
// 
//

#include "opencv2/opencv.hpp"
#include <vector>
#include <numeric>
#include <string>
#include <iostream>
#include <random>
#include <cmath>
#include <tuple>
#include <filesystem>

// NxNxN pixel size
constexpr int N = 32;

// Where to save images
const std::string path = "./32x32x32/";

// file name consist of prefix and the layer number
const std::string file_prefix = "layer_";

// No checks are done to ensure that OpenCV's imwrite() supports file extension.
const std::string file_ext = ".png";

// Sigma parameter of Gaussian kernel used for finding position of largest void/cluster
static constexpr float SIGMA = 1.4;

// Size of filter.
static constexpr int FILTER_SIZE = 17;

// How to initialize random generator.
// "use_random_device = false" will make reproducible noise by seeding random generator with 0
//constexpr bool use_random_device = true;
constexpr bool use_random_device = false;

// Reporting frequency. No need to change this.
//constexpr int REPORT_INTERVAL = -1; // for no reporting
constexpr int REPORT_INTERVAL = 50;    // how often will progress be updated

// In the original paper, for the initial phase authors choose 10% of points. For 3D this seems excessively high.
// Additional heuristic, no regular cubic, fcc, hcp or bcc lattice should be possible ( != n^3, 2*n^3, 4*n^3 )
constexpr int INITIAL_COUNT = (6 * 6 * 6 + 7 * 7 * 7) / 2;

static void intro()
{
    std::cout << "Void-and-Cluster Method for Generating 3D Dither Arrays\n";
    std::cout << "Generating: " << N << "x" << N << "x" << N << " texture\n\n";
}

// percentage==100 will finish up progress output and reset static variables
static void report_progress(int percentage)
{
    static const std::string prefix  = "Progress [";
    static const std::string infix   = "] : ";
    static const std::string postfix = "%";
    static const std::vector<char> arr_progress_wheel = { '\\','|','/','-' };
    static int   wheel_idx = -1;

    static int progress_counter = 0;
    static int last_percentage_reported = -1;


    if (REPORT_INTERVAL <= 0)
        return;

    ++progress_counter;
    if ( progress_counter % REPORT_INTERVAL != 0  &&  percentage != 100 )
        return;


    ++wheel_idx;
    wheel_idx %= arr_progress_wheel.size();
    const char progress_wheel = arr_progress_wheel[wheel_idx];

    if (last_percentage_reported == percentage)
    {
        std::cout << progress_wheel << '\b';
    }
    else if (percentage == 100)  // Finish reporting and reset static variables
    {
        std::string output = prefix + 'X' + infix + std::to_string(percentage) + postfix;
        std::cout << '\r' << output << std::endl;
        percentage = -1;
        progress_counter = -1;
        wheel_idx = -1;
    }
    else
    {
        std::string output = prefix + progress_wheel + infix + std::to_string(percentage) + postfix;
        std::cout << '\r' << output << '\r' << prefix;
    }
    std::cout.flush();

    last_percentage_reported = percentage;
}

static void report_progress_unfinished(int percentage)
{
    report_progress(std::min(percentage, 99));
}

static void report_progress_finished()
{
    report_progress(100);
}


using T3 = std::tuple<int, int, int>;

static void mod(int& a, const int d)
{
    a %= d;
    a += d;
    a %= d;
}

const T3 operator+(const T3& a, const T3& b)
{
    return { std::get<0>(a) + std::get<0>(b), 
             std::get<1>(a) + std::get<1>(b), 
             std::get<2>(a) + std::get<2>(b) };
}

const T3 operator-(const T3& a, const T3& b)
{
    return { std::get<0>(a) - std::get<0>(b), 
             std::get<1>(a) - std::get<1>(b), 
             std::get<2>(a) - std::get<2>(b) };
}

class Matrix3D
{
public:
    Matrix3D(int d0, int d1, int d2) :_d0(d0), _d1(d1), _d2(d2), _d01(d0* d1), _mat3D(d0* d1* d2), _size(d0* d1* d2) {};

    int size()  const               { return _size; };
    int dim0()  const               { return _d0; };
    int dim1()  const               { return _d1; };
    int dim2()  const               { return _d2; };

    //Mat to_mat(int layer) { return Mat(_d1, _d0, CV_32FC1, _arr.data() + layer * _d01); };
    cv::Mat to_mat(int layer) const
    {
        const std::vector<float> _arr_copy(_mat3D.begin() + layer * _d01, _mat3D.begin() + (layer + 1) * _d01);
        cv::Mat cv_mat(_arr_copy, true);
        cv_mat=cv_mat.reshape(1, _d1);
        return cv_mat;
    }

    float& at(const int idx)        { return _mat3D[idx]; };
    float& at(const T3& t3)         { return _mat3D[T3_to_idx(t3)]; };
    float get(const T3& t3) const   { return _mat3D[T3_to_idx(t3)]; };


protected:
    int T3_to_idx(const T3& t3) const
    {
        int i0 = std::get<0>(t3);
        int i1 = std::get<1>(t3);
        int i2 = std::get<2>(t3);
        mod(i0, _d0);
        mod(i1, _d1);
        mod(i2, _d2);
        return i0 + i1 * _d0 + i2 * _d01;
    }
    T3 idx_to_T3(const int idx) const
    {
        return { idx % _d0, (idx % _d01) / _d0, idx / _d01 };
    }
private:
    const int _d0, _d1, _d2;
    const int _d01;
    const int _size;
    std::vector<float> _mat3D;
};


static Matrix3D GaussianMatrix(int size, float sigma);


unsigned int seed()
{
    std::random_device rd;
    return use_random_device ? rd() : 0;
}

class Matrix3D_w_void_and_cluster_tracking : public Matrix3D
{
public:
    Matrix3D_w_void_and_cluster_tracking(int d0, int d1, int d2): 
        Matrix3D(d0, d1, d2),
        weights(d0,d1,d2),
        filter{ GaussianMatrix(FILTER_SIZE, SIGMA) }
    {
        small_randomization();
        void_initialization();
    };

    void set_pixel(const T3& t3, float value)
    {
        if (at(t3) > 0)
            throw std::runtime_error("already set");

        at(t3) = value;
        add_to_cluster(t3);
        conv_at(t3);
    }
    void reset_pixel(const T3& t3)
    {
        at(t3) = 0;
        add_to_void(t3);
        deconv_at(t3);
    }

    void remove_tracking(const T3& t3)
    {
        const int idx = T3_to_idx(t3);
        _track_void.erase({ weights.at(idx), idx });
        _track_cluster.erase({ weights.at(idx), idx });
    }
    void cluster_tracking_off()
    {
        _cluster_tracking_is_on = false;
        _track_cluster.clear();
    }

    const T3   max_void()    const { return idx_to_T3(_track_void.cbegin()->second); };
    const T3   max_cluster() const { return idx_to_T3(_track_cluster.crbegin()->second); };


protected:
    Matrix3D weights;

    const Matrix3D filter;

    std::set<std::pair<float, int>> _track_void;
    std::set<std::pair<float, int>> _track_cluster;
    void add_to_void(const T3& t3)
    {
        const int idx = T3_to_idx(t3);
        _track_cluster.erase({ weights.at(idx), idx });
        _track_void.insert({ weights.at(idx), idx });
    }
    void add_to_cluster(const T3& t3)
    {

        const int idx = T3_to_idx(t3);
        auto was_tracked = _track_void.erase({ weights.at(idx), idx });
        if (was_tracked && _cluster_tracking_is_on)
            _track_cluster.insert({ weights.at(idx), idx });
    }
    void update(const T3& t3, float value)
    {
        int idx = T3_to_idx(t3);
        if (at(idx) != 0)
        {
            auto was_tracked = _track_cluster.erase({ weights.at(idx), idx });
            weights.at(idx) += value;
            if (was_tracked && _cluster_tracking_is_on)
                _track_cluster.insert({weights.at(idx), idx});
        }
        else
        {
            auto was_tracked = _track_void.erase({ weights.at(idx), idx });
            weights.at(idx) += value;
            if (was_tracked)
                _track_void.insert({ weights.at(idx), idx });

        }
    };

    void conv_at(const T3& r)
    {
        const T3 center(filter.dim0() / 2, filter.dim1() / 2, filter.dim2() / 2);
        for (int g2 = 0; g2 < filter.dim2(); ++g2)
        for (int g1 = 0; g1 < filter.dim1(); ++g1)
        for (int g0 = 0; g0 < filter.dim0(); ++g0)
        {
            const T3 g(g0, g1, g2);
            filter.get(g);
            update(r + g - center, filter.get(g));
        }
    }
    void deconv_at(const T3& r)
    {
        const T3 center(filter.dim0() / 2, filter.dim1() / 2, filter.dim2() / 2);
        for (int g2 = 0; g2 < filter.dim2(); ++g2)
        for (int g1 = 0; g1 < filter.dim1(); ++g1)
        for (int g0 = 0; g0 < filter.dim0(); ++g0)
        {
            const T3 g(g0, g1, g2);
            update(r + g - center, -filter.get(g));
        }
    }


    bool _cluster_tracking_is_on{ true };


    void small_randomization()
    {
        float EPS = 1e-7;

        std::mt19937 gen(seed());
        std::uniform_real_distribution<> distr(0, EPS);

        for (int i2 = 0; i2 < weights.dim2(); ++i2)
            for (int i1 = 0; i1 < weights.dim1(); ++i1)
                for (int i0 = 0; i0 < weights.dim0(); ++i0)
                    weights.at({ i0,i1,i2 }) = distr(gen);
    }

    void void_initialization()
    {
        for (int i2 = 0; i2 < weights.dim2(); ++i2)
            for (int i1 = 0; i1 < weights.dim1(); ++i1)
                for (int i0 = 0; i0 < weights.dim0(); ++i0)
                    add_to_void({ i0,i1,i2 });
    }

};

static void show(const Matrix3D& m, std::string window_name = "Layer");

static int dist2(int i0, int i1, int i2)
{
    return (i0 * i0 + i1 * i1 + i2 * i2);
}

static Matrix3D GaussianMatrix(int size, float sigma)
{
    assert(size % 2 == 1);

    Matrix3D g(size,size, size);

    const int center0 = g.dim0() / 2;
    const int center1 = g.dim1() / 2;
    const int center2 = g.dim2() / 2;

    const float inv_sigma2 = 1 / (2 * sigma * sigma);
    for (int i2 = 0; i2 < g.dim2(); ++i2)
    for (int i1 = 0; i1 < g.dim1(); ++i1)
    for (int i0 = 0; i0 < g.dim0(); ++i0)
        g.at({ i0, i1, i2 }) = exp(-dist2(i0 - center0, i1 - center1, i2 - center2) * inv_sigma2);

    return g;
};

static void initial_bitmap(Matrix3D_w_void_and_cluster_tracking& mat3d, int count)
{
    std::mt19937 gen(seed());
    std::uniform_int_distribution<> distr0(0, mat3d.dim0() - 1);
    std::uniform_int_distribution<> distr1(0, mat3d.dim1() - 1);
    std::uniform_int_distribution<> distr2(0, mat3d.dim2() - 1);

    while (count > 0)
    {
        const int i0 = distr0(gen);
        const int i1 = distr1(gen);
        const int i2 = distr2(gen);
        const T3 r(i0, i1, i2);
        if (mat3d.get(r) == 0)
        {
            --count;
            mat3d.set_pixel({ i0, i1, i2 }, 1);
        }
        report_progress_unfinished(0);
    }
}

static void reorder_bitmap(Matrix3D_w_void_and_cluster_tracking& mat3d)
{
    while (true)
    {
        const auto t_max = mat3d.max_cluster();
        mat3d.reset_pixel(t_max);

        const auto t_min = mat3d.max_void();
        mat3d.set_pixel(t_min, 1);

        report_progress_unfinished(0);

        if (t_min == t_max)
            break;
    }
}

static void rank_initial_bitmap(Matrix3D_w_void_and_cluster_tracking& mat3d, int count)
{
    while (count > 0)
    {
        --count;
        
        const auto t_max = mat3d.max_cluster();
        mat3d.reset_pixel(t_max);
        mat3d.set_pixel(t_max, (float)count / mat3d.size());
        mat3d.remove_tracking(t_max);
        report_progress_unfinished(0);
    }
}

static void phase_1(Matrix3D_w_void_and_cluster_tracking& mat3d, int count)
{
    initial_bitmap(mat3d, count);
    reorder_bitmap(mat3d);
    rank_initial_bitmap(mat3d, count);
}

// No need for separate phase 1 and phase 2
// Minimum void 
static void phase_2_and_3(Matrix3D_w_void_and_cluster_tracking& mat3d, int count)
{
    mat3d.cluster_tracking_off();
    for (; count < mat3d.size(); ++count)
    {
        const auto t_min = mat3d.max_void();
        mat3d.set_pixel(t_min, (float)count / mat3d.size());

        report_progress_unfinished(100*count/mat3d.size());
    }
    report_progress_finished();
}

static void show(const Matrix3D& m, std::string window_name)
{
    constexpr char ESC = 27;
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    for (int layer = 0; layer < m.dim2(); ++layer)
    {
        imshow(window_name, m.to_mat(layer));
        char ch;
        while ((ch=cv::waitKey())<0);
        if (ch == ESC)
            break;
    }
}


static void save(const Matrix3D& mat3d, std::string path)
{
    if (!std::filesystem::exists(path))
        std::filesystem::create_directory(path);

    for (int layer = 0; layer < mat3d.dim2(); ++layer)
    {
        cv::Mat mat_float = mat3d.to_mat(layer);
        cv::Mat mat_uchar;
        mat_float.convertTo(mat_uchar,  CV_8UC1, 256);
        cv::imwrite(path + file_prefix + std::to_string(layer) + file_ext, mat_uchar);
    }

    std::cout << "Files saved in: " << path << "\n";
}

int main()
{    
    intro();

    Matrix3D_w_void_and_cluster_tracking mat3d(N, N, N);
    phase_1(mat3d, INITIAL_COUNT);
    phase_2_and_3(mat3d, INITIAL_COUNT);

    try 
    { save(mat3d, path); }
    catch (const cv::Exception& ex)
    { std::cout << "Exception while saving layers: " << ex.what() << std::endl; }

    show(mat3d);

    return 0;
}
