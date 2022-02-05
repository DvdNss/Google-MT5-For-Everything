<!-- PROJECT LOGO -->
<br />
<p align="center">
<h3 align="center">Google MT5 For Everything</h3>

<!-- ABOUT THE PROJECT -->

## About The Project

Google MT5 Transformers models of any size for everything that is input-output NLP using `PyTorch-Lightning`
, `Hugging Face`, `Cuda` and `Streamlit`.

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <ul>
      <li><a href="#structure">Structure</a></li>
      <li><a href="#example">Example</a></li>
    </ul>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- GETTING STARTED -->

## Getting Started

### Installation

1. Clone the repo

```shell
git clone https://github.com/DvdNss/Google-MT5-For-Everything
```

2. Install requirements

```shell
pip install -r requirements.txt
```

<!-- USAGE EXAMPLES -->

## Usage

### Structure

* `data/`: folder containing train/valid files.
* `model/`: folder containing models.
* `resource/`: folder containing repo's imgs.
* `source/`: folder containing source files.
    * `datamodule.py`: data module script.
    * `inference.py`: inference script.
    * `mt5.py`: model script.
    * `train.py`: training script.
* `tokenizer/`: folder containing tokenizer.
* `LICENSE`
* `README.md`
* `requirements.txt`

### Example

1. Build `train.tsv` and `valid.tsv` files, with each having 2 columns: one for inputs and one for outputs, separated
   by `\t`. Inputs must be in the format `task: input`. See examples below.

|Inputs|Outputs|
|:---:|:---:|
| translate: What is your name?| Quel est ton nom? |
| paraphrase: I hate spiders| I dislike spiders |

2. Train a model (see `train.py` script).

```python
from source.datamodule import DataModule
from pytorch_lightning import seed_everything, Trainer
from source.mt5 import MT5

seed_everything(42)

data = DataModule(
    train_file='data/train.tsv',
    valid_file='data/valid.tsv',
    inputs_col='inputs',
    outputs_col='outputs',
    input_max_length=512,
    output_max_length=128,
    batch_size=12,
    added_tokens=['<hl>', '<sep>']
)

model = MT5(model_name_or_path='google/mt5-small', learning_rate=5e-4)

Trainer(max_epochs=10, gpus=1, default_root_dir='model/').fit(model=model, datamodule=data)

# Models will be saved in model/lightning_logs every epoch.
```

3. Use a model for inference (see `inference.py` script).
```python
from transformers import AutoTokenizer

from source.mt5 import MT5

# Loading model and tokenizer
model = MT5.load_from_checkpoint('path_to_checkpoint.ckpt').eval().cuda()
model.tokenizer = AutoTokenizer.from_pretrained('tokenizer', use_fast=True)

inputs = ['question: Who is the French president?  context: Emmanuel Macron is the French president. ']

# Prediction
prediction = model.predict(inputs=inputs)

print(prediction) # --> Emmanuel Macron
```

4. Use Streamlit web app after updating `path_to_checkpoint` in `app.py` with your model path.
```shell
streamlit run app.py
```

<!-- CONTACT -->

## Contact

David NAISSE - [@LinkedIn](https://www.linkedin.com/in/davidnaisse/) - private.david.naisse@gmail.com

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[contributors-url]: https://github.com/Sunwaee/PROJECT_NAME/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[forks-url]: https://github.com/Sunwaee/PROJECT_NAME/network/members

[stars-shield]: https://img.shields.io/github/stars/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[stars-url]: https://github.com/Sunwaee/PROJECT_NAME/stargazers

[issues-shield]: https://img.shields.io/github/issues/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[issues-url]: https://github.com/Sunwaee/PROJECT_NAME/issues

[license-shield]: https://img.shields.io/github/license/sunwaee/PROJECT_NAME.svg?style=for-the-badge

[license-url]: https://github.com/Sunwaee/PROJECT_NAME/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://www.linkedin.com/in/davidnaisse/