# CoRT: Contrastive Rhetorical Tagging

[[Competition Organizer]](https://aida.kisti.re.kr/contest/main/main.do)
[[Problem]](https://aida.kisti.re.kr/contest/main/problem/PROB_000000000000017/detail.do)

The model is built for AI/ML modeling competition hosting from [KISTI](https://www.kisti.re.kr) (Korea Institute of Science and Technology Information).
The main problem of this model is classifying sentences from research papers written in Korean that had been tagged based on rhetorical meaning.

<p align="center">
  <img alt="website" src="https://user-images.githubusercontent.com/5837620/199668919-8a486b94-17ed-4563-95c1-95171cd0c6be.png" width="738px">
</p>

### Problem Solving

The problem have following hierarchical categories.

- Research purpose
  - Problem definition
  - Hypothesis
  - Technology definition
- Research method
  - Suggestion
  - The target data
  - Data processing
  - Theory / models
- Research result
  - Performance / effects
  - Follow-up research

To solve the problem effectively, I have decided to train the model in **Contrastive Learning** manner.
You can use following Pre-trained models: **KorSci-BERT**, **KorSci-ELECTRA**, and **other BERT, ELECTRA, RoBERTa based models from Hugging Face.**

I have used `klue/roberta-base` for additional pre-trained model.

##### Supervised Contrastive Learning (The best in my case)

[[arXiv]](https://arxiv.org/abs/2004.11362) - Supervised Contrastive Learning

The classic contrastive learning is Self-supervised Learning, the model can classify between different objects, but struggling with classifying objects in same label.<br>
In the paper, they suggest supervised-manner learning when you have labels.

I've used contrastive loss from the paper, and Pre-training and Fine-tuning separation.<br>
1. Perform Pre-training in representation learning manner.
2. To perform Fine-tuning, cut off the representation projection layer and attach new classifier layers.

This gave me significant improvement on performance and speed of converge.

##### Margin-based Contrastive Learning

[[arXiv]](https://arxiv.org/abs/2104.08812) - Contrastive Out-of-Distribution Detection for Pretrained Transformers

Contrastive Representation Learning is powerful enough, but pushing all of labels each other may not be that easy.<br>
Maximizing margin between representations is very helpful on clarifying decision boundaries between representations.<br>
Although the paper suggest this for out-of-distribution problem, but experimenting clarifying decision boundaries in other tasks is reasonable.

##### Hierarchical Contrastive Learning

[[arXiv]](https://arxiv.org/abs/2204.13207) - Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework

Since the dataset have sub-depth categories, I thought the model can learn about relationships between top-level categories, and between sub-depth categories at the same time.<br>
The paper suggests training strategy, by pulling together in the same-level categories and pulling stronger when the level is lower and lower.

### Prerequisites

All prerequisites must be up-to-date.
W&B is always required to run pre-training and fine-tuning scripts.
Requiring to use Python 3.8 or above and CUDA 11.4 or above.

Install following main packages by manually, or use `requirements.txt`
```bash
- tensorflow
- tensorflow_addons
- torch
- transformers
- scikit-learn
- pandas
- wandb  # for parameter tracking
- konlpy  # for mecab
- soynlp  # for text normalization
- rich  # for text highlighting
- flask  # for middleware api
```
```bash
pip install -r requirements.txt
```

### Pre-training

W&B Sweeps configurations are available in `./sweeps` directory.<br>
Run automatic hyperparameter tuning by (for example) `wandb sweep ./sweeps/pretraining_supervised.yaml`<br>
And run `wandb agent "{entity_name}/CoRT Pre-training/{sweep_id}"`

To find out how to prepare Pre-trained backbones for Pre-training, read [Pre-trained Backbones README](https://github.com/OrigamiDream/CoRT/blob/main/cort/pretrained/README.md)

Use `build_pretraining_data.py` to create a pre-training dataset from raw texts.
It has the following arguments:
- `--filepath`: Location of raw texts dump that is available at [KISTI](https://doi.org/10.23057/36).
- `--model_name`: Model name to be used as Pre-trained backbones. `korscibert` and `korscielectra` is available by default.
- `--output_dir`: Destination directory path to write out the tfrecords.
- `--korscibert_vocab`: Location of KorSci-BERT vocabulary file. (optional)
- `--korscielectra_vocab`: Location of KorSci-ELECTRA vocabulary file. (optional)
- `--num_processes`: Parallelize tokenization across multi processes. (4 as default)
- `--num_k_fold`: Number of K-Fold splits. (10 as default)
- `--test_size`: Rate of testing dataset. (0.0 as default)
- `--seed`: Seed of random state. (42 as default)

Use `run_pretraining.py` to pre-train the backbone model in representation learning manner.
It has the following arguments:
- `--gpu`: GPU to be utilized for training. ('all' as default, must be int otherwise)
- `--batch_size`: Size of the mini-batch. (64 as default)
- `--learning_rate`: Learning rate. (1e-3 as default)
- `--lr_fn`: Learning rate scheduler type. ('cosine_decay' as default. 'constant', 'cosine_decay', 'polynomial_decay', 'linear_decay' is available)
- `--weight_decay`: Rate of weight decay. (1e-6 as default)
- `--warmup_rate`: Rate of learning rate warmup on beginning. (0.06 as default. the total warmup steps is `int(num_train_steps * warmup_rate)`)
- `--repr_size`: Size of representation projection layer units. (1024 as default)
- `--gradient_accumulation_steps`: Multiplier for gradient accumulation. (1 as default)
- `--model_name`: Model name to be used as Pre-trained backbones.
- `--num_train_steps`: Total number of training steps. (10000 as default)
- `--loss_base`: Name of loss function for contrastive learning. ('margin' as default. 'margin', 'supervised' and 'hierarchical' is available)

The Pre-training takes 3 ~ 4 hours to complete on `NVIDIA A100`

### Fine-tuning

When pre-training is completed, all checkpoints would be located in `pretraining-checkpoints/{wandb_run_id}`

Use `run_finetuning.py` to fine-tune the pre-trained models.
It has the following arguments:
- `--gpu`: GPU to be utilized for training. ('all' as default, must be int otherwise)
- `--batch_size`: Size of the mini-batch. (64 as default)
- `--learning_rate`: Learning rate. (1e-3 as default)
- `--lr_fn`: Learning rate scheduler type. ('cosine_decay' as default. 'constant', 'cosine_decay', 'polynomial_decay', 'linear_decay' is available)
- `--weight_decay`: Rate of weight decay. (1e-6 as default. I recommend to use 0 when fine-tune)
- `--warmup_rate`: Rate of learning rate warmup on beginning. (0.06 as default. the total warmup steps is `int(epochs * steps_per_epoch * warmup_rate)`)
- `--repr_size`: Size of classifier dense layer. (1024 as default)
- `--model_name`: Model name to beu sed as Pre-trained backbones.
- `--pretraining_run_name`: W&B Run ID in `pretraining-checkpoints`. The pre-trained checkpoint model must be same with `--model_name` model.
- `--epochs`: Number of training epochs. (10 as default)
- `--repr_act`: Activation function name to be used after classifier dense layer. ('tanh' as default. 'none', and other name of activations supported from TensorFlow is available)
- `--loss_base`: Name of loss function for contrastive learning. ('margin' as default. 'margin', 'supervised' and 'hierarchical' is available)
- `--restore_checkpoint`: Name of checkpoint file. (`None` as default. I recommend 'latest' when fine-tune)
- `--repr_classifier`: Type of classification head. ('seq_cls' as default. 'seq_cls' and 'bi_lstm' is available)
- `--repr_preact`: Boolean to use pre-activation when activating representation logits. (`True` as default)
- `--train_at_once`: Boolean when you want to train the model from scratch without pre-training. (`False` as default)
- `--repr_finetune`: Boolean when you want to fine-tune the model with additional Representation Learning. (`False` as default) 
- `--include_sections`: Boolean when you want to use 'representation logits of sections' on label representation logits. (`False` as default. `--repr_finetune True` is required for this)

### Inference

Use `run_inference.py` to perform inference on fine-tuned models.
It has the following arguments:
- `--checkpoint_path`: Location of trained model checkpoint. (Required when gRPC server is not provided)
- `--model_name`: Name of pre-trained models. (One of korscibert, korscielectra, huggingface models is allowed)
- `--tfrecord_path`: Location of TFRecord file for inference. {model_name} is a placeholder.
- `--repr_classifier`: Name of classification head for classifier. (One of 'seq_cls' and 'bi_lstm' is allowed)
- `--repr_act`: Name of activation function for representation. (One of 'tanh' and 'gelu' is allowed)
- `--concat_hidden_states`: Number of hidden states to concatenate. (1 as default)
- `--batch_size`: Number of batch size. (64 as default)
- `--max_position_embeddings`: Number of maximum position embeddings. (512 as default)
- `--repr_size`: Number of representation dense units. (1024 as default)
- `--num_labels`: Number of labels. (9 as default)
- `--grpc_server`: Address to TFServing gRPC API endpoint. Specify this argument when gRPC API is available. (`None` as default)
- `--model_spec_name`: Name of model spec. ('cort' as default)
- `--signature_name`: Name of signature of SavedModel ('serving_default' as default)

Perform inference for metrics by (for example) `python run_inference.py --checkpoint_path ./finetuning-checkpoints/wandb_run_id/ckpt-0 --tfrecord_path ./data/tfrecords/{model_name}/valid.fold-1-of-10.tfrecord --concat_hidden_states 2 --repr_act tanh --repr_classifier bi_lstm --repr_size 1024`.<br>
`--concat_hidden_states`, `--repr_act`, `--repr_classifier`, `--repr_size` must be same with configurations that used for fine-tuned model's architecture. 

### Serving

CoRT supports [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) on Docker, use `configure_docker_image.py` to prepare components for Docker container. 
It has the following arguments:
- `--checkpoint_path`: Location of trained model checkpoint. (Required)
- `--saved_model_dir`: Location of SavedModel to be stored. ('./models' as default)
- `--model_spec_name`: Name of model spec. ('cort' as default)
- `--model_spec_version`: Version of model spec. ('1' as default)
- `--signature_name`: Name of signature of SavedModel ('serving_default' as default)
- `--model_name`: Name of pre-trained models. (One of korscibert, korscielectra, huggingface models is allowed)
- `--tfrecord_path`: Location of TFRecord file for warmup requests. {model_name} is a placeholder.
- `--num_warmup_requests`: Number of warmup requests. Pass 0 to skip (10 as default)
- `--repr_classifier`: Name of classification head for classifier. (One of 'seq_cls' and 'bi_lstm' is allowed)
- `--repr_act`: Name of activation function for representation. (One of 'tanh' and 'gelu' is allowed)
- `--concat_hidden_states`: Number of hidden states to concatenate. (1 as default)
- `--repr_size`: Number of representation dense units. (1024 as default)
- `--num_labels`: Number of labels. (9 as default)

Once configuring is done, run following commands to build and run Docker container.
```
docker build -t cort/serving:latest .
docker run -d -p 8500:8500 --name cort-grpc-server cort/serving
```

Intermediate API middleware is written in Flask. Use `run_flask_middleware.py` to open a HTTP server that communicates with gRPC backend directly. It has the following arguments:
- `--host`: Listening address for Flask server ('0.0.0.0' as default)
- `--port`: Number of port for Flask server (8080 as default)
- `--grpc_server`: Address to TFServing gRPC API endpoint. ('localhost:8500' as default)
- `--model_name`: Name of pre-trained models. (One of korscibert, korscielectra, huggingface models is allowed)
- `--model_spec_name`: Name of model spec. ('cort' as default)
- `--signature_name`: Name of signature of SavedModel ('serving_default' as default)

Use `POST http://127.0.0.1:8080/predict` to request prediction over HTTP protocol.
```http request
POST http://127.0.0.1:8080/predict
Content-Type: application/json

{"sentence": "<sentence>"}
```

For people who is unfamiliar with this, the middleware is also providing static website. Visit `http://127.0.0.1:8080/site` to try out easily.

### Performance

[LAN (Label Attention Network)](https://aida.kisti.re.kr/gallery/17) has been proposed in [2021 KISTI AI/ML Competition](https://aida.kisti.re.kr/notice/7).<br>
[Sentence Concat](https://koreascience.kr/article/CFKO202130060700830.pdf) and [Encoder Concat](https://koreascience.kr/article/CFKO202130060700830.pdf) have been proposed by Changwon National Univ. and KISTI Researchers

| Model                            | Macro F1-score | Accuracy  |
|----------------------------------|----------------|-----------|
| Sentence Concat (KLUE BERT base) | 70.85          | 88.77     |
| Encoder Concat (KLUE BERT base)  | 71.91          | 88.59     |
| LAN (KorSci-BERT)                | 89.95          | 89.76     |
| LAN (KLUE RoBERTA base)          | 89.77          | 89.85     |
| **CoRT (KLUE RoBERTA base)**     | **90.50**      | 90.17     |
| **CoRT (KorSci-BERT)**           | 90.42          | **90.25** |

CoRT shows better performance on overall scores comparing with baseline models despite its smaller model architecture.

### Acknowledgement

CoRT was created with GPU support from the [**KISTI National Supercomputing Center (KSC) Neuron**](https://www.ksc.re.kr/ggspcpt/neuron) free trial.
Also, 2 NVIDIA A100 GPUs have been used for Pre-training, and 2 NVIDIA V100 GPUs have been used for Fine-tuning.

### Notes

I don't recommend to use KorSci-ELECTRA because of too high `[UNK]` token rate (about 85.2%).

| Model             | Number of [UNK] | Total Tokens | [UNK] Rate   |
|-------------------|-----------------|--------------|--------------|
| klue/roberta-base | 2,734           | 9,269,131    | 0.000295     |
| KorSci-BERT       | 14,237          | 9,077,386    | 0.001568     |
| KorSci-ELECTRA    | 7,345,917       | 8,621,489    | **0.852047** |

### Citation

If you use this code for research, please cite:
```bibtex
@misc{CoRT2022,
    author = {OrigamiDream},
    title = {CoRT: Contrastive Rhetorical Tagging},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url(https://github.com/OrigamiDream/CoRT)}
}
```

### References
- [Khosla., "Supervised Contrastive Learning", 2020](https://arxiv.org/abs/2004.11362)
- [Zhou., "Contrastive Out-of-Distribution Detection for Pretrained Transformers", 2021](https://arxiv.org/abs/2104.08812)
- [Zhang., "Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework", 2022](https://arxiv.org/abs/2204.13207)
- [Seong., "Rhetorical Sentence Classification Using Context Information", 2021](https://koreascience.kr/article/CFKO202130060700830.pdf)
- [Kim., "Fine-grained Named Entity Recognition using Hierarchical Label Embedding", 2021](http://www.koreascience.or.kr/article/CFKO202130060679826.pdf)