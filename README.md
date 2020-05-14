# Heuristic Algorithms for Application Mapping on mesh-based Network-on-Chip Architectures

This repository contains the following contemporary application mapping strategies implemented as part of my masters thesis at university:

* Random Mapping
* Simulated Annealing
* Integer Linear Programming
* Genetic Algorithm
* Particle Swarm Optimisation
* Artificial Bee Colony
* CastNet

## Usage

To map an application:

```bash
python3 MAPPING_ALGORITHM.py APPLICATION.txt **params
```

Where `MAPPING_ALGORITHM` is one of:
* `RandomMapping.py`
* `GeneticMapping.py`
* `SimulatedAnnealing.py`
* `ILPMapping.py`
* `CastNetMapping.py`
* `PSOMapping.py`
* `ABCMapping.py`

Depending on the chosen algorithm, parameters may be required such as `population size` and `cut off`.

For some examples of `APPLICATION.txt`, please refer to these [example communication task graphs](https://data.mendeley.com/datasets/4fycv9td56/2)