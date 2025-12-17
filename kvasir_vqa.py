from datasets import load_dataset
from tqdm import tqdm
import os

ds = load_dataset("SimulaMet-HOST/Kvasir-VQA")

d_path =r"C:\COMP9800\Dataset\Kvasir-vqa" #existing folder where you want to save images and metadata.csv
os.makedirs(d_path, exist_ok=True)

df = ds['raw'].select_columns(['source', 'question', 'answer', 'img_id']).to_pandas()
print(df.info())
# df.to_csv(f"{d_path}/metadata.csv", index=False)


# os.makedirs(f"{d_path}/images", exist_ok=True)

# for i, row in tqdm(df.groupby('img_id').nth(0).iterrows()): # for images
#   image = ds['raw'][i]['image'].save(f"{d_path}/images/{row['img_id']}.jpg")

df['answer_len'] = df['answer'].apply(lambda x: len(x) )
print(df['answer_len'].describe())