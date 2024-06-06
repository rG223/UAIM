# An Unbounded Archive-Based Inverse Model in Evolutionary Multi-objective Optimization
![Image](./UAIM.svg)
The overview of the UAIM.

This paper has been accepted by 18th International Conference on Parallel Problem Solving from Nature (PPSN 2024). And This project is a pytorch implementation of An Unbounded Archive-Based Inverse Model in Evolutionary Multi-objective Optimization.

## Installation
### Requirements
Our provide the packages file of our environment (requirement.txt), you can using the following command to download the environment:
- pip install -r requirements.txt

## Parameters
- pop_size: The population size of EAs. 
- archive: Whether using UARM (unbounded archive) (1: Using unbounded archive to train inverse model, 0: Do not use unbounded archive to train inverse model)
- mode: Whether using replacement mechanism (1/0)
## Training
```
cd /projects/UAIM
python main.py --problem_name 'dtlz7' --archive 1 --mode 1 --pop_size 55
```
