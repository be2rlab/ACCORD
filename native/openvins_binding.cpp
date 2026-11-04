// OpenVINS (ov_msckf::VioManager, unmodified) for Python: measurements in, the state out.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include <opencv2/opencv.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterHelper.h"
#include "utils/quat_ops.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace py = pybind11;
using namespace ov_msckf;

static std::vector<double> to_vec(const Eigen::VectorXd& v, int a, int b) {
  return std::vector<double>(v.data() + a, v.data() + b);
}

class Openvins {
public:
  explicit Openvins(const std::string& config_path) {
    auto parser = std::make_shared<ov_core::YamlParser>(config_path);
    VioManagerOptions params;
    params.print_and_load(parser);
    if (!parser->successful()) {
      throw std::runtime_error("YamlParser failed to load config: " + config_path);
    }
    vio_ = std::make_shared<VioManager>(params);
  }

  void feed_imu(double t, py::array_t<double> wm, py::array_t<double> am) {
    if (wm.ndim() != 1 || wm.shape(0) != 3 || am.ndim() != 1 || am.shape(0) != 3) {
      throw std::invalid_argument("feed_imu: wm and am must be length-3 arrays");
    }
    auto w = wm.unchecked<1>();
    auto a = am.unchecked<1>();
    ov_core::ImuData m;
    m.timestamp = t;
    m.wm << w(0), w(1), w(2);
    m.am << a(0), a(1), a(2);
    py::gil_scoped_release release;
    vio_->feed_measurement_imu(m);
  }

  void feed_camera(double t, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img, int cam_id) {
    auto b = img.request();
    if (b.ndim != 2) {
      throw std::invalid_argument("feed_camera expects a 2-D grayscale uint8 image");
    }
    cv::Mat gray(static_cast<int>(b.shape[0]), static_cast<int>(b.shape[1]), CV_8UC1, b.ptr);
    ov_core::CameraData cam;
    cam.timestamp = t;
    cam.sensor_ids.push_back(cam_id);
    cam.images.push_back(gray.clone());
    cam.masks.push_back(cv::Mat::zeros(gray.rows, gray.cols, CV_8UC1));
    py::gil_scoped_release release;
    vio_->feed_measurement_camera(cam);
  }

  bool initialized() { return vio_->initialized(); }

  // A zero-velocity pseudo-measurement on the IMU's velocity (standard deviation sigma, m/s), through OpenVINS's own
  // EKF update.
  void zero_velocity_update(double sigma) {
    auto st = vio_->get_state();
    std::vector<std::shared_ptr<ov_type::Type>> order = {st->_imu->v()};
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd res = -st->_imu->vel();
    Eigen::MatrixXd R = sigma * sigma * Eigen::MatrixXd::Identity(3, 3);
    py::gil_scoped_release release;
    StateHelper::EKFUpdate(st, order, H, res, R);
  }

  // At rest, as OpenVINS's ZUPT measures it (UpdaterZeroVelocity, its standard form, the same equations and weights):
  // over the IMU samples given (times t, gyroscope wm, accelerometer am), the gyroscope reads no rotation and the
  // accelerometer gravity in the current attitude, which constrains the attitude and both biases; then the velocity
  // pseudo-measurement above. Unlike OpenVINS's ZUPT, the camera's frames are processed as always.
  void zero_motion_update(py::array_t<double, py::array::c_style | py::array::forcecast> t,
                          py::array_t<double, py::array::c_style | py::array::forcecast> wm,
                          py::array_t<double, py::array::c_style | py::array::forcecast> am, double sigma_w, double sigma_a,
                          double gravity_mag, double noise_multiplier, double sigma_v) {
    auto T = t.unchecked<1>();
    auto W = wm.unchecked<2>();
    auto A = am.unchecked<2>();
    const int n = static_cast<int>(T.shape(0));
    if (W.shape(0) != n || A.shape(0) != n || W.shape(1) != 3 || A.shape(1) != 3) {
      throw std::invalid_argument("zero_motion_update: t (N), wm and am (N, 3)");
    }
    auto st = vio_->get_state();
    if (n >= 2) {
      std::vector<std::shared_ptr<ov_type::Type>> order = {st->_imu->q(), st->_imu->bg(), st->_imu->ba()};
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6 * (n - 1), 9);
      Eigen::VectorXd res = Eigen::VectorXd::Zero(6 * (n - 1));
      const Eigen::Vector3d gravity(0., 0., gravity_mag);
      const Eigen::Matrix3d Dw = State::Dm(st->_options.imu_model, st->_calib_imu_dw->value());
      const Eigen::Matrix3d Da = State::Dm(st->_options.imu_model, st->_calib_imu_da->value());
      const Eigen::Matrix3d Tg = State::Tg(st->_calib_imu_tg->value());
      const Eigen::Matrix3d R_jacob = st->_options.do_fej ? st->_imu->Rot_fej() : st->_imu->Rot();
      for (int i = 0; i < n - 1; i++) {
        const double dt = T(i + 1) - T(i);
        const Eigen::Vector3d a(A(i, 0), A(i, 1), A(i, 2)), w(W(i, 0), W(i, 1), W(i, 2));
        const Eigen::Vector3d a_hat = st->_calib_imu_ACCtoIMU->Rot() * Da * (a - st->_imu->bias_a());
        const Eigen::Vector3d w_hat = st->_calib_imu_GYROtoIMU->Rot() * Dw * (w - st->_imu->bias_g() - Tg * a_hat);
        const double w_omega = std::sqrt(dt) / sigma_w, w_accel = std::sqrt(dt) / sigma_a;
        res.segment<3>(6 * i) = -w_omega * w_hat;
        res.segment<3>(6 * i + 3) = -w_accel * (a_hat - st->_imu->Rot() * gravity);
        H.block<3, 3>(6 * i, 3) = -w_omega * Eigen::Matrix3d::Identity();
        H.block<3, 3>(6 * i + 3, 0) = -w_accel * ov_core::skew_x(R_jacob * gravity);
        H.block<3, 3>(6 * i + 3, 6) = -w_accel * Eigen::Matrix3d::Identity();
      }
      UpdaterHelper::measurement_compress_inplace(H, res);
      if (H.rows() > 0) {
        Eigen::MatrixXd R = noise_multiplier * Eigen::MatrixXd::Identity(res.rows(), res.rows());
        py::gil_scoped_release release;
        StateHelper::EKFUpdate(st, order, H, res, R);
      }
    }
    zero_velocity_update(sigma_v);
  }

  // Time, IMU state (JPL quaternion global to IMU, position, velocity, biases) and the camera-IMU calibration.
  py::dict get_state() {
    auto st = vio_->get_state();
    py::dict d;
    d["t"] = static_cast<double>(st->_timestamp);
    Eigen::VectorXd imu = st->_imu->value();
    d["q_GtoI"] = to_vec(imu, 0, 4);
    d["p"] = to_vec(imu, 4, 7);
    d["v"] = to_vec(imu, 7, 10);
    d["bg"] = to_vec(imu, 10, 13);
    d["ba"] = to_vec(imu, 13, 16);
    if (st->_calib_IMUtoCAM.count(0)) {
      Eigen::VectorXd e = st->_calib_IMUtoCAM.at(0)->value();
      d["q_ItoC"] = to_vec(e, 0, 4);
      d["p_IinC"] = to_vec(e, 4, 7);
    }
    // The camera's intrinsics (fx fy cx cy and the distortion) and time offset, as the filter holds them.
    if (st->_cam_intrinsics.count(0)) {
      Eigen::VectorXd k = st->_cam_intrinsics.at(0)->value();
      d["intrinsics"] = to_vec(k, 0, static_cast<int>(k.size()));
    }
    d["dt_CAMtoIMU"] = static_cast<double>(st->_calib_dt_CAMtoIMU->value()(0));
    d["slam_features"] = static_cast<int>(st->_features_SLAM.size());
    d["msckf_features"] = static_cast<int>(vio_->get_good_features_MSCKF().size());   // used by the last frame's update
    return d;
  }

private:
  std::shared_ptr<VioManager> vio_;
};

PYBIND11_MODULE(openvins_ext, m) {
  m.doc() = "OpenVINS (ov_msckf::VioManager) for Accord";
  // OpenVINS reads no print level from its configuration file: INFO unless set here (diagnosis: DEBUG prints the
  // features each update used, per frame).
  m.def("set_print_level", [](const std::string& level) { ov_core::Printer::setPrintLevel(level); }, py::arg("level"));
  py::class_<Openvins>(m, "VioManager")
      .def(py::init<const std::string&>(), py::arg("config_path"))
      .def("feed_imu", &Openvins::feed_imu, py::arg("t"), py::arg("wm"), py::arg("am"))
      .def("feed_camera", &Openvins::feed_camera, py::arg("t"), py::arg("img"), py::arg("cam_id") = 0)
      .def("initialized", &Openvins::initialized)
      .def("zero_motion_update", &Openvins::zero_motion_update, py::arg("t"), py::arg("wm"), py::arg("am"),
           py::arg("sigma_w"), py::arg("sigma_a"), py::arg("gravity_mag"), py::arg("noise_multiplier"), py::arg("sigma_v"))
      .def("get_state", &Openvins::get_state);
}
