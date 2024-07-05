# PASTEL

To run the code, follow these steps:

1. Place the dataset inside the ```data/datasets``` folder. Dataset files must be in a csv format and have the following columns: ```title,text,objective,article_md5```.
   
   * title: title of the news article (string).
   * text: body of text of the news article (string).
   * objective: misinformation label, either 0 (non-misinformation) or 1 (misinformation) (integer).
   * article_id: a unique identifier for each article (string).

2. Install the dependencies.
   ```shell
   conda install -n pastel python==3.10.4
   conda activate pastel
   pip install -r requirements.txt
   ``` 

3. Extract the credibility signals for the desired dataset.
   ```shell
   python3 scripts/prompt.py --dataset mydataset --model_size=70
   ```
   Supposing your csv is in ```data/datasets/mydataset.csv```.
   The model will save each processed article in a cache file in ```data/cache/mydataset.jsonl```. Once the entire dataset is processed, proceed to the next step. You can choose a smaller model (7 or 13) if you wish.

4. Consolidate the processed cache into a csv file.
   ```shell
   python3 scripts/consolidate.py --dataset mydataset
   ```
   Given that your cache file is in ```data/cache/mydataset.jsonl```, you will find a new csv file in ```data/signals/mydataset.csv```.

5. Train the weakly supervised label model using the extracted signals.
   ```shell
   python3 scripts/train_ws.py --dataset mydataset --model_size=70
   ```
   Given that the processed signals are in ```data/signals/mydataset.csv```, there should now be a file named ```mydataset_metrics.json``` in the root folder with the average metrics computed with 10-fold cross validation.

## Other experiments

### Fine-tuning the LLaMa-FT baseline:

1. After installing the code dependencies, run:
   ```shell
   python3 scripts/finetune.py
   ```
