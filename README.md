# GPT2DST Model Card

Last updated: Last updated: April 2022


Inspired by [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993), we’re providing some accompanying information about the GPT-2 family of models we're releasing.

## Model Details.

Our model is designed for Dialogue State Tracking(DST) which could recognize the dialogue acts for the entire dialogue. This model is built upon [HuggingFace model](https://huggingface.co/gpt2) with only fine-tuning on a small in-domain dataset after 200,000 bathes on the [MultiWOZ 2.1](https://github.com/budzianowski/multiwoz/tree/master/data) dataset. 

### Model type

Language model

## To Use:

We provide the ckpt of our model after 200,000 bathes. To inference on your data, you could use the follow command:
```
    python gpt2-dst.py --hyp_dir OUTPUT \
    --test_data DST_DATA --args DST_ARGS --checkpoint $CHECKPOINT -vv
```

where the hyp_dir is the folder hypothesis files are to be saved, DST_DATA is where you save you test data, DST_ARGS is the configuration of decoding, checkpoint is the trained model you want to inference on and '-vv' is the logging level you may want to use. You could also replace '-vv' with '-v' or '--quiet' as you like. We provide the default trained model and args configuration in the file, you could also alter the configuration as you like. Notice that the configuration file contains information like maximum sequence length and experiment name. You may need to alter this setting if you would like to inference our model on your own dataset.
 
## To Train:
To train you own model on your data, you could use the follow command:
```
python $BDIR/train-gpt2-dst.py CT \
 --train_data TRAIN_DATA --dev_data DEV_DATA --args TRAIN_ARGS -vv
```

CT is the pre-trained model, you could also delete it which means that you would like to tran a flat start model. TRAIN_DATA and DEV_DATA is where you save you train and development data, TRAIN_ARGS is the configuration of training, checkpoint, model settings and so on. You may need to alter the configuration file to you need. For example, the 'experiment_name' in this file will be the folder containing your model. Once the training process has been done, you could also test you model performance in the previous commands. Remember to alter the checkpoint filepath to you needs.

## Further Work
This model is a simple version of implementing DST models without further fine-tuning on the hyperparameters. The next step will be fine-tuning the model on more datasets. This model structure could be used for training other tasks like Natural Language Understanding (NLU) as well. You just need to alter the dataset as well as the configuration accordingly.
