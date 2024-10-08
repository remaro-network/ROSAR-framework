TODO: upload implementation

ROSAR: RObust Sonar Adversarial Re-training

## Setup

1. Clone the repository:
`git clone --recursive https://github.com/remaro-network/ROSAR-framework.git`

2. Setup conda environment:

```
# Change current working directory to ROSAR repository
cd ROSAR-framework

# Remove the old environment, if necessary.
conda deactivate; conda env remove --name ROSAR-framework

# install all dependents into the alpha-beta-crown environment
conda env create -f alpha-beta-CROWN/complete_verifier/environment.yaml --name ROSAR-framework

# activate the environment
conda activate ROSAR-framework
```

## Example usage

1. Generate robustness property
```
# Assuming that the current working directory is the main directory of ROSAR
python generate_P1.py ./data/Compressed_SSS400.png 0 0.10
```

2. Copy the safety property into the alpha-beta-CROWN tool
```
# Copy the instances.csv file
cp instances.csv ./alpha-beta-CROWN/complete_verifier/models/yolox/

# Copy the vnnlib file
cp ./vnnlib/Compressed_SSS400_perturbed_bbox_0_delta_0.1.vnnlib ./alpha-beta-CROWN/complete_verifier/models/yolox/vnnlib/
```

## Acknowledgements

This work is part of the Reliable AI for Marine Robotics (REMARO) Project. For more info, please visit: https://remaro.eu/

[<img src="media/remaro-right-1024.png">](https://remaro.eu/)

<a href="https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-2020_en">
    <img align="left" height="60" alt="EU Flag" src="https://remaro.eu/wp-content/uploads/2020/09/flag_yellow_low.jpg">
</a>

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 956200.
