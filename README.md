# PEDPF <img src="https://icai.ai/wp-content/uploads/2020/01/AIRLabAmsterdam-10-6-gecomprimeerd-transparant.png" width="300" alt="Airlab Amsterdam" align="right"> #

_Parameter Efficient Deep Probabilistic Forecasting_ (PEDPF) is a repository containing code to run experiments for several deep learning based probabilistic forecasting methods. For more details, see [our paper](https://arxiv.org/abs/2112.02905).

### Reproduce paper's experiments ###
First, you need to download the necessary data files.
* [UCI Electricity](https://archive.ics.uci.edu/ml/machine-learning-databases/00321/) Download and extract `LD2011_2014.txt` to the folder `data\uci_electricity\` (create if necessary).
* [UCI Traffic](https://archive.ics.uci.edu/ml/machine-learning-databases/00204/) Download and extract to the folder `data\uci_traffic\`. Run `create_df.py` to create the dataset from the source files.
* [Kaggle Favorita](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data) Download and extract all files to the folder `data\kaggle_favorita` and run `prepare_favorita_v3.py` (NB: there is some code missing here, this needs to be fixed).
* [Kaggle Webtraffic](https://www.kaggle.com/c/web-traffic-time-series-forecasting/data) Download and extract all files to the folder `data\kaggle_webtraffic` and run `prepare_webtraffic.py`.

Then, run `experiments\train.py` for the paper's main results. This will sequentially run all the experiments as listed in the `experiments\{dataset_name}\experiments_{dataset_name}.csv` file. Hence, to change parameters or create more experiments, it is easiest to adjust this `.csv` file.

The other experiments can be run using the variants of `experiments\train_{}.py`. Note that some of these variants require installing additional dependencies as well as creating new folders manually.

### Reference ###
[Olivier Sprangers](mailto:o.r.sprangers@uva.nl), Sebastian Schelter, Maarten de Rijke. [Parameter Efficient Deep Probabilistic Forecasting](https://arxiv.org/abs/2112.02905). Accepted as journal paper to [International Journal of Forecasting](https://www.journals.elsevier.com/international-journal-of-forecasting).

### License ###
This project is licensed under the terms of the [Apache 2.0 license](https://github.com/elephaint/pgbm/blob/main/LICENSE).

### Acknowledgements ###
This project was developed by [Airlab Amsterdam](https://icai.ai/airlab/).
