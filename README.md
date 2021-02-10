## BioBERET-MCNN
This repository provides the code for fune-tuning in Named Entity Recognition (NER) tasks.
Please refer to our paper [BioBERT-MCNN: A semantic integration approach for biomedical named entity recognition.](https://XXX) for more details

## Pre-trained weights
Pre-training was based on the [BERT code](https://github.com/google-research/bert) provided by google and Pre-trained weights used [BioBERT Base V1.0 (+PubMed 200K + PMC 270K)](https://github.com/dmis-lab/biobert) provided by BioBERT.

##  Datasets
We used 8 BioNER datasets (BC4CHEMD,BioNLP09,BioNLP11D,NCBI,Linnaeus,BC2GM,BC5CDR-disease and BC5CDR-chem) for experiments in our paper. All the datasets can get in the ner_data directory.
The program will automatically process the original data if there is no tfrecord file in the data directory.  

## Installation
Sections below describe the installation and the fine-tuning process of our model based on Tensorflow 1.52 (python version = 3.6).
To fine-tune BioBERT-MCNN, you need to download the  [pre-trained weights of BioBERT](https://github.com/dmis-lab/biobert)
After downloading the pre-trained weights, `requirements.txt` to install our model as follows:
```bash
$ cd BioBERT-MCNN; pip install -r requirements.txt
```

## Fine-tune 
For fine-tuning in NER tasks, you can get the exmaple bash and run the bash as follow:
```bash
$ cd BioBERT-MCNN
$ ./fine-tune
```
The meaning of the parameters in bash is as follows:
* model_config_path: set the pre-trained model configuration file path.
* init_checkpoint: set the path of pre-trained weights .
* model_dir: set the weights path of fine-tuning for saving.
* vocab_file: set the path of vocabulary text.
* train_batch_size: set the number of training batch size.
* eval_batch_size: set the number of evaluation batch size.
* task: set the task name.
* do_train: weather to do training, default True.
* do_eval: weather to do evaluation, default True.
* do_predict: weather to do prediction, default True.
* data_dir: set the path for read data.
* learning_rate: set the number of learning rate.
* trainable_layer: set the trainable layer of BioBERT, default 12
* trainable_layer: set the trainable layer of BioBERT, default 12
* label_mode: set the label mode for fine-tuning, the value should be BL or WPL.
* train_epoch: set the number of train_epoch.
* result_dir: set the saving path of evaluation results.
* no_cnn: set weather to use MCNN for fine-tuning, default True denoting to not use MCNN.


 


