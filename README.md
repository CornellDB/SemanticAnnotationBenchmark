# SemanticAnnotationBenchmark

This repository is a collection of code and files used to run experiments and data collection for the semantic annotation.

## Data Labelling Interface

This folder contains code for our streamlit based data annotation tool. In order to set it up on a new instance, we will need to set up multiple things.
Here are the high level steps:

1. Setup Pinecone database, S3 bucket, EC2 instance.
2. Modify/create a .env secrets file in the root folder of this repository with the necessary Pinecone API keys, Azure Openai API keys and AWS keys.
```
AZURE_OPENAI_API_KEY={}
PINECONE_API_KEY={}
AWS_ACCESS_KEY_ID={}
AWS_SECRET_ACCESS_KEY={}
```
3. Upload datasets to S3 bucket in the same folder paths as stated in `current_dataset.csv`. You can find the zip file
4. Run `python3 semantic_search.py` to upload semantic concepts to Pinecone database. This script will first generate embeddings of deduplicated concepts in `dbpedia_concepts2.txt` and `wikidata_concepts2.txt`. It will then proceed to upsert those concepts to a Pinecone vector database. If you would like to use a local vector database, you will have to modify the streamlit code accordingly.

To run the application and deploy it on EC2 instance, here are the next few steps:
1. SSH into your ec2 instance and clone this repository.
2. Set up an elastic IP on the AWS dashboard and assign it to the EC2 instance so that the IP address of the app will remain the same.
3. Run `cd SemanticAnnotationBenchmark/data_labelling_interface`
4. Run `pip3 install -r requirements.txt`
5. Run ` streamlit run app.py`
6. Exit the tmux window and check that your app is now deployed.


This folder also contains the `data_scripts/` folder which contains scripts to run postprocessing. Download the latest labels from the s3 bucket and run `python3 postprocessing.py --label-filename {label_filename}`. This will generate a new csv file called postprocessed_labels.csv.
The `data_scripts/` folder also contains the `eval.py` script that runs evaluation of outputs on the benchmarks.

## GPT Experiments

This folder contains jupyter notebooks based of [Github repository of the 'Column Type Annotation using ChatGPT' paper](https://github.com/wbsg-uni-mannheim/TabAnnGPT), with modified prompts to experiment performance of predictions without the restriction of classes.
To run the experiments, you will first have to set up the conda environment using `conda env create -f llm.yml`

Predictions were done on both the sotab dataset used in the ['Column Type Annotation using ChatGPT' paper](https://arxiv.org/abs/2306.00745) paper as well as the dataset used for the [C2 paper](https://arxiv.org/abs/2012.08594).

1. `Prompt-table-experiments.ipynb`
This notebook loads the original sotab data used in the paper and is able to run multiple different prompts templates that we defined. The main new prompt that was tested was the "semantic_concept_template" prompt. It also allows you to change the table format to be column major. 

Lastly, the notebook does evaluation. There is a modified evaluation method that uses bertscore to find the closest match between the predicted open-world concept by GPT and the ground truth defined classes. This creates a csv file in the predictions folder and an additional human evaluated step is required to analyze if the match is accurate.

2. `Prompt-table-experiments-c2.ipynb`
This notebook loads c2 data and is able to run and evaluate different prompt templates, similar to the other notebook.