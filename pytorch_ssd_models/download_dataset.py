import urllib.request
import pandas as pd
from tqdm import tqdm

dataset=pd.read_csv('newest.csv')
dataset=dataset.to_numpy()
i=0
for url in tqdm(dataset[:,1]):
#     print(url)
#     print(type(url))
    i=i+1
    img_name=url.split('/')[-1]
    if(i<int(0.8*dataset.shape[0])):
        
        urllib.request.urlretrieve(url,"./data/train/"+img_name)
        
    else:
        urllib.request.urlretrieve(url,"./data/test/"+img_name)