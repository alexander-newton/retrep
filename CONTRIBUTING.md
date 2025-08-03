# Contributing

Create a new folder for each new replication.
Where possible split data from code. 
Raw data from replication packages should be saved in a raw data folder.
The location of this folder can be edited in your personal `config.yaml` file.
Code used in replication should be saved in this repository.
In addition, any notes on the replication process can be saved in this repository.

Each code folder should correspond to one paper.
The code folder should contain everything needed to install dependencies for that paper and the code required for replication.
The code folder must not contain any data.

Intermediate data can be stored in a separate folder, noted in `config.yaml`.
If a replication takes a lot of computational time, consider making intermediate datasets near to completion to reduce the burden for co-authors.

For each replication, the target output is a `xxx.json` file stored in the `target` folder noted in `config.yaml`. 

```json
{
  "paper_id": "001",
  "tables": [
    {
      "table_id": "3.2",
      "column": 1, #which column of the table is relevant
      "binary": 0, #indicates whether treatment is binary
      "model": "log-linear", #log-linear, exponential
      "elasticity": 1, #1 if elasticity, 0 if semi-elasticity
      "FEs": [4,5,6], #which variables are fixed effects (zero-indexed)
      "IVs": [8], #which variables are instruments (zero-indexed)
      "interest": [0], #which variables are most of interest
      "y": [1.2,4.5,...], #the y variable column,
      "X": [[1.6,2.3,...],[2.3,45.6,...]],#the X matrix,
    },
    {
      ...
    },
  ]
}


```
