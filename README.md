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

## Acknowledgements

This work is part of the Reliable AI for Marine Robotics (REMARO) Project. For more info, please visit: https://remaro.eu/

[<img src="media/remaro-right-1024.png">](https://remaro.eu/)

<a href="https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-2020_en">
    <img align="left" height="60" alt="EU Flag" src="https://remaro.eu/wp-content/uploads/2020/09/flag_yellow_low.jpg">
</a>

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 956200.
