# Learning to Incentivize Others

This is the code for experiments in the paper [Learning to Incentivize Other Learning Agents](https://arxiv.org/abs/2006.06051). Baselines are included.


## Setup

- Python 3.6
- Tensorflow >= 1.12
- OpenAI Gym == 0.10.9
- Clone and `pip install` [Sequential Social Dilemma](https://github.com/011235813/sequential_social_dilemma_games), which is a fork from the [original](https://github.com/eugenevinitsky/sequential_social_dilemma_games) open-source implementation.
- Clone and `pip install` [LOLA](https://github.com/alshedivat/lola) if you wish to run this baseline.
- Clone this repository and run `$ pip install -e .` from the root.


## Navigation

* `alg/` - Implementation of LIO and PG/AC baselines
* `env/` - Implementation of the Escape Room game and wrappers around the SSD environment.
* `results/` - Results of training will be stored in subfolders here. Each independent training run will create a subfolder that contains the final Tensorflow model, and reward log files. For example, 5 parallel independent training runs would create `results/cleanup/10x10_lio_0`,...,`results/cleanup/10x10_lio_4` (depending on configurable strings in config files).
* `utils/` - Utility methods


## Examples

### Train LIO on Escape Room

* Set config values in `alg/config_room_lio.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_multiprocess.py lio er`. Default settings conduct 5 parallel runs with different seeds.
* For a single run, execute `$ python train_lio.py er`.

### Train LIO on Cleanup

* Set config values in `alg/config_ssd_lio.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_multiprocess.py lio ssd`.
* For a single run, execute `$ python train_ssd.py`.


## License

See [LICENSE](LICENSE).

SPDX-License-Identifier: MIT
