"""
utils/json_utils.py

Utility module for creating JSON output files for replication results.
This module can be imported and used across different paper replications.
"""

import json
import numpy as np
import os
import yaml
from pathlib import Path


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path('config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Default configuration
        return {
            'rawdata': './rawdata',
            'intermediatedata': './intermediate_data',
            'finaldata': './final_data',
            'output': './output'
        }


class ReplicationJSONBuilder:
    """Class to build and save JSON output for replication results"""
    
    def __init__(self, paper_id):
        """
        Initialize JSON builder for a specific paper.
        
        Args:
            paper_id (str): The paper ID (e.g., "004")
        """
        self.paper_id = paper_id
        self.tables = []
        self.config = load_config()
        self.output_folder = self.config.get('output', './output')
    
    def add_table(self, table_id, column, y, X, interest_idx, 
                  fe=None, fe_indices=None, z=None, iv_indices=None, 
                  binary=0, model="log-linear", elasticity=False):
        """
        Add a table/regression result to the JSON output.
        
        Args:
            table_id (str): Table identifier (e.g., "3.B")
            column (int): Column number in the table
            y (np.array): Dependent variable values
            X (np.array): Independent variables (without FE)
            interest_idx (int or list): Index(es) of variable(s) of interest
            fe (np.array, optional): Fixed effects variables
            fe_indices (list, optional): Indices where FEs appear in full X matrix
            z (np.array, optional): Instrument variables
            iv_indices (list, optional): Indices of instruments in X matrix
            binary (int): 1 if treatment is binary, 0 otherwise
            model (str): Model type ("log-linear", "linear", "exponential")
            elasticity (bool): Whether computing elasticity
        
        Returns:
            dict: The created table entry
        """
        
        # Ensure interest_idx is a list
        if isinstance(interest_idx, int):
            interest_idx = [interest_idx]
        
        # Combine X and FE for full X matrix if FE exists
        if fe is not None:
            X_full = np.hstack([X.reshape(-1, 1) if X.ndim == 1 else X, fe])
        else:
            X_full = X.reshape(-1, 1) if X.ndim == 1 else X
        
        # Add instruments to X matrix if they exist
        if z is not None:
            X_full = np.hstack([X_full, z.reshape(-1, 1) if z.ndim == 1 else z])
        
        # Create the table entry according to the specification
        entry = {
            "table_id": table_id,
            "column": column,
            "binary": binary,
            "model": model,
            "elasticity": 1 if elasticity else 0,
            "FEs": fe_indices if fe_indices else [],
            "IVs": iv_indices if iv_indices else [],
            "interest": interest_idx,
            "y": y.flatten().tolist(),
            "X": [X_full[:, i].tolist() for i in range(X_full.shape[1])]
        }
        
        self.tables.append(entry)
        return entry
    
    def save(self, filename=None, verbose=True):
        """
        Save the JSON output to file.
        
        Args:
            filename (str, optional): Output filename. If None, uses paper_id.json
            verbose (bool): Whether to print confirmation message
        
        Returns:
            str: Path to the saved JSON file
        """
        if filename is None:
            filename = f"{self.paper_id}.json"
        
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)
        
        output_path = os.path.join(self.output_folder, filename)
        
        json_output = {
            "paper_id": self.paper_id,
            "tables": self.tables
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        if verbose:
            print(f"JSON output saved to: {output_path}")
        
        return output_path
    
    def get_json(self):
        """
        Get the JSON structure without saving to file.
        
        Returns:
            dict: The complete JSON structure
        """
        return {
            "paper_id": self.paper_id,
            "tables": self.tables
        }
    
    def clear(self):
        """Clear all tables from the builder"""
        self.tables = []
    
    def summary(self):
        """Print a summary of the current JSON structure"""
        print(f"Paper ID: {self.paper_id}")
        print(f"Number of tables: {len(self.tables)}")
        for i, table in enumerate(self.tables, 1):
            print(f"  Table {i}: {table['table_id']}, Column {table['column']}")
            print(f"    - Model: {table['model']}")
            print(f"    - N obs: {len(table['y'])}")
            print(f"    - N variables: {len(table['X'])}")
            print(f"    - Fixed Effects indices: {table['FEs']}")
            print(f"    - Interest variable indices: {table['interest']}")


# Convenience function for quick single-table JSON creation
def create_json_output(paper_id, table_id, column, y, X, interest_idx,
                      fe=None, fe_indices=None, z=None, iv_indices=None,
                      binary=0, model="log-linear", elasticity=False,
                      save=True, filename=None):
    """
    Quick function to create and save JSON for a single table.
    
    Args:
        paper_id (str): Paper ID
        table_id (str): Table identifier
        column (int): Column number
        y (np.array): Dependent variable
        X (np.array): Independent variables
        interest_idx (int or list): Index of variable of interest
        fe (np.array, optional): Fixed effects
        fe_indices (list, optional): FE indices in full X matrix
        z (np.array, optional): Instruments
        iv_indices (list, optional): IV indices
        binary (int): 1 if binary treatment
        model (str): Model type
        elasticity (bool): Whether elasticity
        save (bool): Whether to save immediately
        filename (str, optional): Output filename
    
    Returns:
        dict or str: JSON structure if save=False, filepath if save=True
    """
    builder = ReplicationJSONBuilder(paper_id)
    builder.add_table(table_id, column, y, X, interest_idx,
                     fe, fe_indices, z, iv_indices, 
                     binary, model, elasticity)
    
    if save:
        return builder.save(filename)
    else:
        return builder.get_json()
