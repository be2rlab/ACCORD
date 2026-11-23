# One-time setup and datasets. Day-to-day use is ./accord.
SHELL := /bin/bash
DATA  ?= data
CUDA_VERSION ?= 12.6.3
TORCH_VERSION ?= 2.13.0
IN_IMAGE := ACCORD_GPUS=none scripts/docker.sh

.PHONY: help setup image openvins binding weights xfeat euroc tumvi ori hilti spires phone_rig test clean

help:
	@echo "make setup        image, OpenVINS, its binding, DA3 weights, XFeat (about an hour)"
	@echo "make image        both images [CUDA_VERSION=12.6.3 TORCH_VERSION=2.13.0; also accepted by setup]"
	@echo "make weights      a Depth Anything 3 checkpoint [MODEL=base|large|giant]"
	@echo "make euroc        EuRoC into data/euroc [SEQS='V1_01_easy V1_02_medium']"
	@echo "make tumvi        TUM-VI rooms into data/tumvi [SEQS='room1 room2']"
	@echo "make ori          ORI sequences into data/ori, converted, with their scan [SEQS='r01 r02']"
	@echo "make hilti        Hilti-Oxford sequences into data/hilti, converted, with their scan [SEQS='exp14 exp18']"
	@echo "make spires       Oxford Spires sequences into data/spires, converted, with their scan [SEQS='cc03 cc05']"
	@echo "make phone_rig    the phone rig dataset, once downloaded: data/rig_android, data/rig_iphone [RELEASE=its folder]"
# 	@echo "make test         the test suite, inside the image"

setup: image openvins binding weights xfeat

image:
	docker build -f docker/Dockerfile.base \
	  --build-arg CUDA_VERSION="$(CUDA_VERSION)" \
	  --build-arg TORCH_VERSION="$(TORCH_VERSION)" \
	  -t accord/standalone-gpu:latest docker/
	docker build -f docker/Dockerfile.dense -t accord/dense-gpu:latest docker/

openvins:
	git submodule update --init thirdparty/open_vins thirdparty/Depth-Anything-3
	scripts/docker.sh bash -c 'cd thirdparty/open_vins && colcon build --packages-up-to ov_msckf --cmake-args -DCMAKE_BUILD_TYPE=Release'

binding:
	scripts/docker.sh native/build.sh

MODEL ?= giant
weights:
	scripts/fetch_da3_weights.sh $(MODEL)

xfeat:
	scripts/fetch_xfeat.sh

euroc: SEQS ?= V1_01_easy V1_02_medium
euroc:
	python3 scripts/fetch_euroc.py --data-root $(DATA) --sequences $(SEQS)

tumvi: SEQS ?= room1
tumvi:
	scripts/fetch_tumvi.sh $(DATA) $(SEQS)

ori: SEQS ?= r01
ori:
	python3 scripts/fetch_ori.py --data-root $(DATA) --sequences $(SEQS)
	for s in $(SEQS); do \
	  $(IN_IMAGE) python3 scripts/convert_ori.py data/ori/$$s && $(IN_IMAGE) python3 scripts/ori_reference.py data/ori/$$s || exit 1; \
	done

hilti: SEQS ?= exp14
hilti:
	python3 scripts/fetch_hilti.py --data-root $(DATA) --sequences $(SEQS)
	[ -d thirdparty/py310_extra/pye57 ] || $(IN_IMAGE) python3 -m pip install -q --target thirdparty/py310_extra pye57
	for s in $(SEQS); do \
	  [ -f data/hilti/$$s/accord/conversion.json ] || $(IN_IMAGE) python3 scripts/convert_hilti.py data/hilti/$$s || exit 1; \
	  [ -f data/hilti/$$s/accord/pointcloud0/data.ply ] || $(IN_IMAGE) python3 scripts/hilti_reference.py data/hilti/$$s || exit 1; \
	done

spires: SEQS ?= cc03
spires:
	python3 scripts/fetch_spires.py --data-root $(DATA) --sequences $(SEQS)
	for s in $(SEQS); do \
	  [ -f data/spires/$$s/accord/conversion.json ] || $(IN_IMAGE) python3 scripts/convert_spires.py data/spires/$$s || exit 1; \
	  [ -f data/spires/$$s/accord/pointcloud0/data.ply ] || $(IN_IMAGE) python3 scripts/spires_reference.py data/spires/$$s || exit 1; \
	done

phone_rig:
	@test -f "$(RELEASE)/sequences.csv" || { echo "make phone_rig RELEASE=<the dataset's folder, which holds sequences.csv>" >&2; exit 2; }
	mkdir -p $(DATA)
	ln -sfn "$(abspath $(RELEASE))" $(DATA)/rig_android
	ln -sfn "$(abspath $(RELEASE))" $(DATA)/rig_iphone

test:
	./accord test

clean:
	rm -rf native/build src/openvins_ext*.so
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
