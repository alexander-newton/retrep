# Contributing

Create a new folder for each new replication in the code folder.
Where possible split data from code. 

Raw data from replication packages should be saved in a raw data folder.
The location of this folder can be edited in your personal `config.yaml` file.
Code used in replication should be saved in this repository.
In addition, any notes on the replication process can be saved in this repository.

Each folder in code should correspond to one paper.
The code folder should contain everything needed to install dependencies for that paper and the code required for replication.
The code folder must not contain any data.

Intermediate data can be stored in a separate folder, noted in `config.yaml`.
If a replication takes a lot of computational time, consider making intermediate datasets near to completion to reduce the burden for co-authors.

For each replication, provide metadata of the form as a dictionary in python script:

```json
{
    'paper_id': '011;
    'table_id': '2',
    'panel_identifier': 'A1_2',
    'model_type': 'log-linear',
    'comments': "Couldn't replicate standard errors"
}


```
