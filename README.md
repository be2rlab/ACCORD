# Accord

Dense metric mapping from one camera and an IMU, online, with loop closure. An OpenVINS filter tracks the camera; Depth Anything 3 gives depth for windows of keyframes; each window is anchored to metric scale by the filter's points and placed as a submap in a pose graph that loop closures correct; at export, the overlaps between submaps correct the map once more.

## Quick start

```bash
git clone --recursive https://github.com/be2rlab/ACCORD.git && cd ACCORD
make setup                         # once, about an hour: the image, OpenVINS, its binding, DA3-Base, XFeat
make euroc SEQS=V1_01_easy         # one EuRoC room
./accord run V1_01_easy --viz      # live, with the viewer; then the map is exported and scored
```

The run prints the viewer's address (http://localhost:9090/...) and ends with its scores; its directory is
`runs/V1_01_easy`. `make help` lists every setup target, `./ACCORD` with no argument every command.

## Setup

Linux, Docker with the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/),
an NVIDIA GPU (6 GB for DA3-Base, 24 GB for DA3-Giant), about 30 GB of disk, and `git`, `make` and Python 3 on the host.
Nothing else is installed on the host: everything runs in the image.

```bash
git clone --recursive https://github.com/be2rlab/ACCORD.git && cd ACCORD
make setup                      # image, OpenVINS, its Python binding, DA3-Base weights, XFeat (about an hour, once)
make weights MODEL=giant        # the paper's model (also: large), from Hugging Face
```

## Data

| dataset | fetch | sequences |
|---|---|---|
| EuRoC (Vicon rooms) | `make euroc SEQS='V1_01_easy V2_01_easy'` | `V1_01_easy` ... `V2_03_difficult` |
| TUM-VI | `make tumvi SEQS='room1 room4'` | `room1` ... `room6` |
| ORI (Oxford Robotics Institute) | `make ori SEQS='r01 r02'` | `r01` ... `r05` |
| Hilti-Oxford | `make hilti SEQS='exp18 exp16 exp21'` | `exp01` ... `exp21` ([config/hilti/README.md](config/hilti/README.md)) |
| Oxford Spires | `make spires SEQS='cc03 cc05'` | `cc02`, `cc03`, `cc05` |
| the phone rig (Note 9 and iPhone) | download it, then `make phone_rig RELEASE=<its folder>` | `vio01` ... `vio13`, as `rig_android` and `rig_iphone` (the Note 9 has no `vio03`) |
| RealSense bag | `scripts/convert_realsense.py` (its docstring) | your recordings, rig in `config/realsense` |
| iPhone (Stray Scanner) | `python3 scripts/convert_iphone.py data/Iphone/ID` (`--rig`, `--turn` for another phone or one held in landscape) | your recordings, rig in `config/iphone` |
| your own camera + IMU | ASL layout in `data/custom/SEQ/mav0` (`cam0/data/<ns>.jpg`, `imu0/data.csv`) | rig in `config/phone` |

ORI, Hilti-Oxford and Oxford Spires come with a laser scan; `make` converts them and builds the reference the map is scored
against. The phone rig's reference is a LiDAR phone's RTAB-Map graph and map, which reach only about 3.5 m from its track:
its maps are scored without their points more than 3 m from the track, and the share left out is reported
(`beyond_reach` in `surface.json`).

## Run, with the viewer

`./accord run SEQ --dataset NAME` runs the system live at sensor rate, then exports the map and scores it. `--viz` opens a
viewer beside the run: open http://localhost:9090/?url=rerun%2Bhttp%3A%2F%2F127.0.0.1%3A9876%2Fproxy (the run prints it).

```bash
./accord run V1_01_easy --viz                                   # EuRoC
./accord run room4 --dataset tumvi --viz                        # TUM-VI
./accord run r01 --dataset ori --rate 0.3 --viz                 # ORI: two-megapixel frames, at 0.3 of real time
./accord run exp18 --dataset hilti --viz                        # Hilti-Oxford
./accord run cc03 --dataset spires --viz                        # Oxford Spires
./accord run vio07 --dataset rig_iphone --viz                   # the phone rig: the iPhone's recording of walk vio07
./accord run vio07 --dataset rig_android --viz                  #                the Note 9's
./accord run NAME --dataset realsense --viz                     # a converted RealSense bag
./accord run NAME --dataset iphone --viz                        # a converted iPhone recording
./accord run NAME --dataset custom --viz                        # your own recording
```

Add `--weights thirdparty/da3_weights/giant` for the paper's model, `--set mapping.online_correction=true` to correct the
map while it is built, `--out RUN` to choose the run directory (default `runs/SEQ`). Every option is in
`python3 scripts/run.py --help`; every setting in [config/accord.yaml](config/accord.yaml).

```bash
./accord view runs/V1_01_easy                     # a finished run, or one still running
./accord view runs/V1_01_easy --save tour.rrd     # a file, to open with `rerun tour.rrd`
./accord map runs/V1_01_easy --no-correction --out runs/V1_01_easy/map_fused.ply   # the map as built
./accord score runs/V1_01_easy
```


## Scoring and ablation

[docs/SCORING.md](docs/SCORING.md): `scripts/score.sh RUN...` scores runs and prints the table.
[docs/ABLATION.md](docs/ABLATION.md): `scripts/ablation.sh ARM...` runs the paper's ablation.

## Reproducing the paper

The paper's runs use [config/accord.yaml](config/accord.yaml) and the rig files of `config/` as they are, with DA3-Giant
(`make weights MODEL=giant`). Beyond them a run is given:

- `--weights thirdparty/da3_weights/giant`, on every dataset; ORI also `--rate 0.3` (its two-megapixel frames);
- nothing per sequence. The update at rest is on for Hilti-Oxford only (`standstill.rigs`), chosen on its development
  sequences, and every rig keeps the IMU noise its file carries;
- the map of the main table is corrected at export (the default); the correction while the map is built
  (`--set mapping.online_correction=true`) is scored as `surface_online.json`, the *live* column;
- three runs per sequence, each on an otherwise idle machine: the mapper drops windows when the GPU is busy, so load
  changes the map.

| results | command |
|---|---|
| main table, every public dataset | `scripts/paper.sh euroc tumvi ori hilti spires` |
| the phone rig, both phones, every recording | `scripts/paper.sh rig_android rig_iphone` |
| cumulative ablation | `scripts/ablation.sh none anchor system` |
| simpler alternatives, reference poses, no loop closure | `scripts/ablation.sh global videpth refposes noloops` |
| depth models | `scripts/ablation.sh base large giant` |
| correction while the map is built | `scripts/ablation.sh online` |

`scripts/summarize.py` prints the mean over sequences of the mean over runs, with the paper's failure rule
([docs/SCORING.md](docs/SCORING.md)). Where the paper differs from the summary: on Hilti-Oxford it scores maps on the three
scanned evaluation sequences (exp18, exp16, exp21) and the trajectory at the surveyed control points where the reference is
sparse (both in each run's reports), and it ran the simpler alternatives once per ORI sequence. Live runs depend on
timing, so a rerun differs a little from the paper's.

## Licence

GPL-3.0, as OpenVINS. The Depth Anything 3 weights have their own: base is Apache-2.0; large and giant are CC BY-NC 4.0.
