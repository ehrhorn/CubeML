# CubeML
==============================

IceCube track reconstruction using machine learning.

Path to i3 files:

`/groups/hep/ehrhorn/files/icecube/hdf5_files/MuonGun_Level2_139008`

HDF5 structure:

```
data/
├── lvl0/
│   ├── transform0/
|   |   └── dataset1
|   |   └── dataset2
│   ├── transform1/
|   |   └── dataset1
|   |   └── dataset2
│   └── transform2/
|       └── dataset1
|       └── dataset2
├── lvl1/
│   ├── transform0/
|   |   └── dataset1
|   |   └── dataset2
│   ├── transform1/
|   |   └── dataset1
|   |   └── dataset2
│   └── transform2/
|       └── dataset1
|       └── dataset2
pred/
└── lvl0/ 
    └── transform1
        └── model1
            └── prediction1
            └── prediction2
```

Note: `transform0` represents the untransformed data.

## Guides til Bjørn

##### Table of Contents
[Kør Powershovel](#powershovel)

[Kør `i3_to_hdf5_calc.py`](#run)

[Plotly](#plotly)

<a name="powershovel"/>

### Kør Powershovel

Stå i projektet rod.
Skift til CubeML conda environment med

`conda activate cubeml`

Opdatér conda pakker (Powershovel skal bruge Streamlit) med:

`conda env update`

Kør

`streamlit run src/visualization/Powershovel.py`

Du er _nødt_ til at stå i projektets rod når du kører Streamlit,
paths i Powershell er ikke intelligente. :(

Der er _nødt_ til at køre den lokalt (Ubuntu/Windows), for den starter en
webserver og en browser, og det vil hep ikke tillade.

Den kan være lidt sløv til at starte op, for den skal hente 11000 events,
men den er quite smooth når man har brugt sliders til at barbere lidt fra.

Den er god til at finde insights om IceCube events!

<a name="run"/>

### Kør `i3_to_hdf5_calc.py`

NOTE!!! Filen hedder nu `i3_to_hdf5_calc.py`, fordi vi udfører en calculation
i IceTray (det gør vi ikke i den anden fil; derfor hedder den
`i3_to_hdf5_no_calc.py`; kom endelig med forslag til bedre navne).

Vær i Bash (_ikke_ fish!!!)

Kør kommandoen

`eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4/setup.sh)`

Gå til din `icerec/build` mappe.

Kør kommandoen

`./env-shell.sh`

Gå til CubeML projektmappen, og derfra til `src/data`.

Kør kommandoen

`ipython`

som åbner in IPython kommandoprompt.

I IPython kommandoprompten kører du

`%run i3_to_hdf5_calc.py`

og så... kører IPython filen...

<a name="plotly"/>

### Plotly

Du finder et Plotly eksempel i `notebooks/1.0-meh-plotly_test.py`.

Vær opmærksom på at Plotly måske får VS Code til at køre langsomt.
Dette kan løses ved at køre i en klassisk Jupyter Notebook, og du kan
eksportere Python filen som notebook ved at trykke `Ctrl+shift+P`
og indtaste `jupyter`.
Her kan du vælge

`Python: Export current Python File as Jupyter Notebook`

hvilket åbner en browser og viser filen i en klassisk notebook.

NOTE!!! Når du kører Plotly ønsker du _ikke_ at bruge det Python environment
IceCube giver os.
Derfor, hvis du allerede har kørt

`eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4/setup.sh)`

er du lige nødt til at logge af hep og på igen.
Dernæst kan du køre kommandoen

`conda env create`

i projektets root mappe (i.e. den hvor `environment.yml` ligger).
Kommandoen laver et nyt Anaconda environment, og installerer de pakker
der er angivet i `environment.yml`; det inkluderer Plotly.

Good luck!

<!-- # Old stuff

## Introduction

This repository can, for now, ingest `i3` files from the IceCube
collaboration, and spit out `hdf5` files.
These can further be ingested and converted to dictionaries
of `Pandas` dataframes.

Ingesting `i3` files requires a working `Icetray` environment;
see the guide on this page.

You'll need your own `i3` files---these are not provided here.

Use `i3_read/i3_to_hdf5.py` for converting `i3` files to `hdf5`.

Use `read_data.py` as a template for reading `hdf5` files and converting
them to dictionaries of `Pandas` dataframes.

## How to install IceCube software

We'll install `icerec`, which includes much of what we want.

I'll assume you're working on NBI's `hep` servers, as these have
`cvmfs` which is needed.

First, navigate to the `icerec` SVN repository
[here](https://code.icecube.wisc.edu/svn/meta-projects/icerec/), and click
on `releases/`.
Choose the last link on the page, as that will be the latest release.
Now, open your terminal and create a folder for `icerec` where
you store your apps.
Inside the `icerec` folder, create a `build` and a `source` folder by doing
`mkdir build source`.
The `build` folder will contain the compiled `icerec` code, while `source`
will contain the `icerec` source code.

Navigate to your recently created `source` folder, and type:

`svn co https://code.icecube.wisc.edu/svn/meta-projects/icerec/releases/V05-02-05/`

Note: The `/V05-02-05/` in the above code _may be different for you_, as it
pertains to the version of `icerec` you're fetching.
Thus, your version may be newer.

This checks out and downloads the code from the SVN repository.

Now run
`eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4/setup.sh)`
This sets up your toolchain (incl. `CMake` and your Python interpreter)
to originate from `cvmfs`.
Note the `py3-v4` bit in the statement; this refers to the version of
Python we'll use from `cvmfs`.
Here, it gets a bit tricky.
You should compare the IceTray date compatibility from
(here)[http://software.icecube.wisc.edu/documentation/info/cvmfs.html]
with the compile date found in the `icerec` readme file (found on SVN).
These dates should match.

Now, in the `icerec/build/` folder, do:
`cmake ../source`
This readies the compilation.
When done, run
`make`
and the compilation proper will start.
If you have access to more CPUs, do
`make -j n`
instead, where n is the number of CPUs you want to use.
Be careful!

The compilation will take some time.

## How to use IceTray

When the software is built, you can use `./env-shell.sh` to activate
IceTray.
Note: you must run 
`eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4/setup.sh)`
first.
This sets up the toolchain needed to use `icerec`.
Now you may run the file `i3_read/i3_to_hdf5.py` to convert `i3` files
to `hdf5`.
Note: you'll probably want to exit the IceTray environment, if you want
to run other Python scripts.
Do this by entering `ctrl+d` once.

While in an active IceTray environment, you can use data-shovel to view
the content of an `i3` file, so you know what keys to extract.
Do this by entering
`dataio-shovel <filename>`
where `<filename>` is the name of an `i3` file.


## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p> -->
