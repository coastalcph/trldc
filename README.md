This repository has a pytorch implementation of hierarchical transformers for long document classification, introduced in our paper:

> Xiang Dai and Ilias Chalkidis and Sune Darkner and Desmond Elliott. 2022. Revisiting Transformer-based Models for Long Document Classification.

Please cite this paper if you use this code. The paper can be found at <a href="https://arxiv.org/abs/2204.06683">ArXiv</a>.

### Data
* sample data can be found at data/sample.json

### Experiments
* sample script can be found at scripts/sample.sh

### Task-adaptive pre-trained models

Models are available at
~~~
wget iang.io/resources/trldc/mimic_roberta_base.zip
wget iang.io/resources/trldc/ecthr_roberta_base.zip

wget iang.io/resources/trldc/mimic_longformer.zip
wget iang.io/resources/trldc/ecthr_longformer.zip

~~~
or using
~~~
from transformers import AutoConfig, AutoTokenizer, AutoModel

config = AutoConfig.from_pretrained("xdai/mimic_longformer_base") # or xdai/mimic_roberta_base
tokenizer = AutoTokenizer.from_pretrained("xdai/mimic_longformer_base")
model = AutoModel.from_pretrained("xdai/mimic_longformer_base") 
~~~
