# SISEC MUS 2018 Analysis and Visualization

## Installation

install the python3.6 requirements using [pipenv](https://docs.pipenv.org/): `pipenv install`

## Usage

### Aggregate Data

To aggregate the data and convert them to various formats we use a [pandas](https://pandas.pydata.org/) data frame.

`python aggregate.py ./EST --out sisec18.pd` generates a pandas data frame with all results from the `./EST` folder.

If you want to compare several estimates, you can just specify them as multiple command line arguments:

```sh
python aggregate.py ./EST1 ./EST2 --out sisec18.pd
```

or use a parent folder and bash wildcards

```sh
python aggregate.py ./ESTIMATES/* --out sisec18.pd
```

### Plots and Statiscial Analysis

[Please see jupyter notebook](museval18-analysis.ipynb)

### SiSEC 18 Paper Plots

t.b.a
