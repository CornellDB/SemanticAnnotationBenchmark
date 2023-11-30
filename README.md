# SemanticAnnotationBenchmark

This repository is a collection of code and files used to run experiments and data collection for the semantic annotation.

## Data Labelling Interface

This folder contains code for our streamlit based data annotation tool. In order to set it up on a new instance, we will need to set up multiple things.
Here are the high level steps:

1. Setup Pinecone database, S3 bucket, EC2 instance.
2. Modify .env secrets file with the necessary Pinecone API keys and AWS keys.
3. Upload datasets to S3 bucket in the same folder paths as stated in `current_dataset.csv`.
4. Run `python3 semantic_search.py` to upload semantic concepts to Pinecone database.

This folder also contains the `data_scripts/` folder which contains scripts to run postprocessing. Download labels from the s3 bucket and run `python3 postprocessing.py --label-filename {label_filename}`. 
The `data_scripts/` folder also contains the `eval.py` script that runs evaluation of outputs on the benchmarks.

## GPT Experiments

This folder contains jupyter notebooks based of https://github.com/wbsg-uni-mannheim/TabAnnGPT, with modified prompts to experiment performance of predictions without the restriction of classes.

Predictions were done on both the sotab dataset used in the https://arxiv.org/abs/2306.00745 paper as well as the dataset used for the C2 paper.