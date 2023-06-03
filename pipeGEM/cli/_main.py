import argparse


def main():
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--gene_data",
                        dest="gene_data_conf_path",
                        metavar="config_file_path")

    main()