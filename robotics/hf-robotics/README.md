# Notes
## 0. Welcome to the Robotics Course
Done

## 1. Course Introduction
Done
- [x] set up git
- [x] set up uv environment

- Birth of both robotics and AI in the 1950s
- [Unimate](https://en.wikipedia.org/wiki/Unimate) as the first industrial robot (1961)
- [ ] Watch later: [Paradigm Shifts - Robot Learning](https://www.youtube.com/watch?v=VEs1QYEgOQo)
- LeRobot is an open-source library for robotics and AI research, providing tools for simulation, control, and learning algorithms.
- [official docs here](https://huggingface.co/docs/lerobot/index)
- Specialized data format: `LeRobotDataset`
- Separation of thinking and doing


### `LeRobotDataset`
- Key aspects
  - multi-modal
  - temporal
  - episodic
  - high dimensional

- Three components
  - tabular data: low-dimensional, high-frequency data (e.g. joint states & actions)
  - visual data: camera data
  - metadata: json-files

- Library supports streaming


## 2. Classical Robotics

