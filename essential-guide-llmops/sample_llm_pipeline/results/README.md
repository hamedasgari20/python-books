---
library_name: peft
license: apache-2.0
base_model: google/flan-t5-small
tags:
- generated_from_trainer
model-index:
- name: stackoverflow-flan-t5-small
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# stackoverflow-flan-t5-small

This model is a fine-tuned version of [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 29.0210

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| No log        | 1.0   | 4    | 29.0594         |
| No log        | 2.0   | 8    | 29.0337         |
| No log        | 3.0   | 12   | 29.0210         |


### Framework versions

- PEFT 0.15.2
- Transformers 4.52.4
- Pytorch 2.7.1+cpu
- Datasets 3.6.0
- Tokenizers 0.21.2