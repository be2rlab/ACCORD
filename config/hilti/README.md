# Hilti-Oxford (Hilti SLAM Challenge 2022) rig

One camera and the IMU of the five-camera Alphasense rig; the lidar and the other cameras are never read.

- `source/calib_3_cam0-1-camchain-imucam.yaml`: the dataset's Kalibr chain, verbatim (`calibration/calibration_files/`
  of the release, CC BY-NC-SA 3.0).
- `kalibr_imucam_chain.yaml`: its **cam0** in this repository's convention: Kalibr ships `T_cam_imu`, OpenVINS reads
  `T_imu_cam`, the inverse (`tests/test_hilti.py` checks it). 720 x 540 equidistant at 40 Hz, `timeshift_cam_imu`
  +1.9 ms. The lens reaches 42 degrees above its centre, so it does not fill the 90 degree square the depth model gets
  from a fisheye; it gets the undistorted image of the same size instead (`accord.images.depth_view`).
- `kalibr_imu_chain.yaml`: the BMI085 at 400 Hz (399.2 Hz by its stamps). Noise densities of the Allan deviation of
  the release's 1.5 h at rest (`calibration/imu_noise_calibration.bag`): accelerometer 1.42e-3 m/s^2/sqrt(Hz),
  gyroscope 2.44e-4 rad/s/sqrt(Hz); bias walks three times the Allan bounds, 5.58e-4 and 1.26e-5.
- `estimator_config.yaml`: the ORI rig's filter settings with `track_frequency: 41` (40 Hz camera) and
  `gravity_mag: 9.81`.
- `downloads.json`: the pinned revision of the Hugging Face release with the size and sha256 of every file, and the
  ground-truth files of each sequence.

Reference (scoring only): surveyed control points, millimetre-accurate, for every sequence, and dense poses at 10 Hz,
accurate to 1-2 cm, for exp14, exp16 and exp18; both are the IMU in the frame of the site's laser scan. Trajectories
are scored against the dense poses where there are some, and at the control points with the dataset's own score
(`accord/eval/control_points.py`) everywhere. Maps are scored against the scan cropped to the sequence
(`scripts/hilti_reference.py`); a sequence with control points only is placed by them, and its support is the run's own
cameras so placed.

Fetch and prepare: `make hilti HILTI='exp14 exp18'` (bags, ground truth, conversion, reference cloud). Run:
`./accord run exp14 --dataset hilti`.
