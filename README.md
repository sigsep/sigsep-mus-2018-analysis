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

### Plots and Statistical Analysis

We provide a [jupyter notebook](museval18-analysis.ipynb) that includes all the results that were used to create the SiSEC evaluation. Also you can run the notebook on [google colab](https://drive.google.com/file/d/1DoGm0WizK_jmgdo1lSVAQRTMESNr6IyO/view?usp=sharing).


### SiSEC 18 Paper Plots

t.b.a

### SiSEC 18 Paper

t.b.a
