

Official implementation of CoPHo (KDD 2026).
=======
## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed

  - Create a environment with python 3.9:
    
    ```conda create -c conda-forge -n copho  python=3.9```
    
  - `conda activate copho`

  - Install graph-tool (https://graph-tool.skewed.de/): 
    
    ```conda install -c conda-forge graph-tool=2.45```
    
  - Check that this line does not return an error:
    
    ```python3 -c 'import graph_tool as gt' ```
    
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
    
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
    
  - Install opacus 1.5.3: 

​	```pip install opacus==1.5.3```

> Note: `opacus` is a lightweight PyTorch library for differentially private training. It is **not** used anywhere in this paper. We require it only because, during the development of CoPHo, some checkpoints were saved (`torch.save`) in an environment where another project was experimenting with differential privacy using `opacus`. As a result, these checkpoints contain class references to `opacus`, and having the package installed prevents deserialization errors when loading the pretrained models.

  - Install other packages using the requirement file: 

    ```cd copho```

    ```pip install -r requirements.txt```

  - Run:

    ```pip install -e .```

  - Navigate to the ./src/analysis/orca directory and compile orca.cpp: 

     ```g++ -O2 -std=c++11 -o orca orca.cpp```

## Quick Reproduction

For convenience, once the environment is set up, CoPHo supports one-click generation and evaluation.

- Generation:

  ```bash
   ./run_all.sh 
  ```

- Evaluation:

```python
python evaluation.py
```

This pipeline reproduces the four community experiments reported in our main table (Table 2, *community*).

After the script finishes, you should obtain results close to (the mean absolute error (MAE) between the ground-truth and the generated graphs, lower is better):

| range   | density | assortativity | transitivity | clustering |
| ------- | ------- | ------------: | -----------: | ---------: |
| 0.6–1.0 | 1.89    |          10.9 |         7.23 |       7.13 |
