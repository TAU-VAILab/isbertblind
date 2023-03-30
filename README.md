# Is BERT Blind? Exploring the Effect of Vision-and-Language Pretraining on Visual Language Understanding

This repository is for the paper

Morris Alper, Michael Fiman, & Hadar Averbuch-Elor (2023). Is BERT Blind? Exploring the Effect of Vision-and-Language Pretraining on Visual Language Understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. ([arXiv link](https://arxiv.org/abs/2303.12513))

For more information, see our project page at https://isbertblind.github.io.

## Code

This repository allows can be used for the following:
- Apply Stroop Probing on a given sentence with different options
- Compare MLM and Stroop Probing for different types of models on various tasks 

###  Stroop Probing
#### Setting up environment
Build the stroop-probing package locally by running the following commands: 
```commandline
# clone into this repo
git clone https://github.com/TAU-VAILab/isbertblind.git
# install the Stroop Probing package locally
pip install -e ./isbertblind
```

#### Probing
An example for usage of Stroop probing:
```python
COLORS = ['red', 'orange', 'yellow', 'green', 'blue', 'black', 'white', 'grey', 'brown']
SENTENCE = 'A MASK colored banana'

from probing import CLIPStroopProbe

clip_sp = CLIPStroopProbe('openai/clip-vit-base-patch32')
scores = clip_sp.score_from_options(SENTENCE, COLORS, as_dict=True)
# scores = {'red': 0.78190374, 'orange': 0.8250252, 'yellow': 0.8816596, 'green': 0.8156868, 
# 'blue': 0.79435384, 'black': 0.79852706, 'white': 0.83922243, 'grey': 0.81859416, 'brown': 0.8265251}

print(f"{SENTENCE.replace('MASK', max(scores, key=scores.get))}")
# A yellow colored banana
```
The following models are supported:
- CLIP: ```CLIPStroopProbe``` - uses huggingface checkpoints supporting ```CLIPModel.from_pretrained()```  
- FLAVA: ```FLAVAStroopProbe``` - uses huggingface checkpoints supporting  ```FlavaModel.from_pretrained()```
- TEXT: ```TextStroopProbe``` - uses huggingface checkpoints supporting  ```AutoModel.from_pretrained()``` which use a pooler output layer 

###  Probing comparison
#### Setting up environment
```commandline
# clone into this repo
git clone https://github.com/TAU-VAILab/isbertblind.git
# install required packages for using this repo
pip install -r requirements.txt
```
#### Tasks

This repository currently supports two types of tasks:
##### choice 
This type of task is used to test association between a given set of objects and a given set of words. For example, this type of task can be used for color or shape association prediction.

Use the config file define experiment setup parameters, set of prompts to test and list of models to use. An example for a task config can be found in the `./configs/shapes.json` file.

This sort of task requires a csv file with the following columns: `["word","gt","options"]`. An example of a dataset for using this type of task, please see the `datasets/shape_association.csv` file.

##### cloze
This type of task is used to solve cloze tasks. For example, this kind of task can be found in the Children’s Book Test (CBT) cloze dataset.

Use the config file define experiment setup parameters, set of words to use as PAD options, and list of models to use. An example for a task config can be found in the `./configs/cbt_v_sample.json` file.

This sort of task requires a csv file with the following columns: `["sentence","gt","options"]`. An example of a dataset for using this type of task, please see the `datasets/cbt_v_sample.csv` file.

Note that since the Children’s Book Test (CBT) Dataset is not ours, we only show a sample of a few examples in this repository. 

## Datasets

The **ShapeIt** dataset of shape associations introduced by our paper is [available at Kaggle](https://www.kaggle.com/datasets/morrisalp/shapeit). The other datasets used in our paper are publicly available and can be accessed at their respective project pages.

Other datasets used for the different VLU and NLU tasks which were used in our paper can be found in the following links:
- [Concreteness](https://github.com/ArtsEngine/concreteness.git)
- [CTD](https://www.kaggle.com/datasets/rtatman/color-terms-dataset)
- [NCD](https://drive.google.com/file/d/1k_UvYzdrHbphW4UcbDb9jWB0ZQIAGEAo/view)
- [Comparative Question Completion](https://github.com/google-research-datasets/comparative-question-completion)
- [Children’s Book Test](https://research.facebook.com/downloads/babi/)
- [Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

## Citation

If you find this code or our data helpful in your research or work, please cite the following paper.
```
@InProceedings{alper2023:is-bert-blind,
    author    = {Morris Alper and Michael Fiman and Hadar Averbuch-Elor},
    title     = {Is BERT Blind? Exploring the Effect of Vision-and-Language Pretraining on Visual Language Understanding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
```