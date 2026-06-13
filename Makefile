REPO_ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
ROS2_WS   := $(REPO_ROOT)/ros2_ws

.PHONY: build sync clean run run-custom install-deps

## ros2_ws を自己完結させてから colcon build
build: sync
	cd $(ROS2_WS) && colcon build --symlink-install

## src/vision_pipeline/ と configs/ を ros2_ws 内に同期
## ros2_ws を別リポジトリに切り出す前にも実行すること
sync:
	rsync -a --delete $(REPO_ROOT)/src/vision_pipeline/ \
		$(ROS2_WS)/src/vision_pipeline_pkg/vision_pipeline/
	rsync -a --delete $(REPO_ROOT)/configs/ \
		$(ROS2_WS)/configs/

## ビルド成果物を削除
clean:
	rm -rf $(ROS2_WS)/build $(ROS2_WS)/install $(ROS2_WS)/log

## recognition_node を起動
run:
	bash -c "source $(ROS2_WS)/install/setup.bash && ros2 launch nedo_vision recognition.launch.py"

## imu_topic / camera_topic を変更して起動
## 例: make run-custom IMU=/your/imu CAMERA=/your/camera
run-custom:
	bash -c "source $(ROS2_WS)/install/setup.bash && \
		ros2 launch nedo_vision recognition.launch.py \
		imu_topic:=$(IMU) camera_topic:=$(CAMERA)"

## ローカル開発用（uv run pytest / pyright を使う場合のみ必要）
install-deps:
	pip install -e $(REPO_ROOT)
