import argparse
from pathlib import Path

import pandas as pd

from ._io import parse_toml_file, load_data
from pipeGEM.data.data import GeneData


def load_gene_data(gene_data_conf):
    inp_params = gene_data_conf["input"]
    input_file_path = inp_params.pop("input_file_path")
    data = load_data(file_path=input_file_path, **inp_params)

    GeneData()



def main():
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--gene_data",
                        dest="gene_data_conf_path",
                        metavar="config_file_path")

    main()