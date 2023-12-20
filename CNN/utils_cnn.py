## Imports
import os
import re
import random
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib.colors import hex2color
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot as plot
import json
from random import shuffle
try:
  from pytorch_lightning.utilities.distributed import rank_zero_only
except ImportError:
  from pytorch_lightning.utilities.rank_zero import rank_zero_only  

###OLD DICTS

lut_colors = {
1   : '#db0e9a',
2   : '#938e7b',
3   : '#f80c00',
4   : '#a97101',
5   : '#1553ae',
6   : '#194a26',
7   : '#46e483',
8   : '#f3a60d',
9   : '#660082',
10  : '#55ff00',
11  : '#fff30d',
12  : '#e4df7c',
13  : '#3de6eb',
14  : '#ffffff',
15  : '#8ab3a0',
16  : '#6b714f',
17  : '#c5dc42',
18  : '#9999ff',
19  : '#000000'}

lut_classes = {
1   : 'building',
2   : 'pervious surface',
3   : 'impervious surface',
4   : 'bare soil',
5   : 'water',
6   : 'coniferous',
7   : 'deciduous',
8   : 'brushwood',
9   : 'vineyard',
10  : 'herbaceous vegetation',
11  : 'agricultural land',
12  : 'plowed land',
13  : 'swimming_pool',
14  : 'snow',
15  : 'clear cut',
16  : 'mixed',
17  : 'ligneous',
18  : 'greenhouse',
19  : 'other'}

### NEW DICTS

new_lut_colors = {
1   : '#194a26',
2   : '#46e483',
3   : '#f3a60d',
4   : '#660082',
5  : '#55ff00',
6  : '#c5dc42',
7  : '#000000'}

new_lut_classes = {
1   : 'coniferous',
2   : 'deciduous',
3   : 'brushwood',
4   : 'vineyard',
5  : 'herbaceous vegetation',
6  : 'ligneous',
7  : 'other'}


def get_data_paths (path, filter):
    for path in Path(path).rglob(filter):
         yield path.resolve().as_posix()



def new_class(old_class_number: int, old_dict: dict = lut_classes, new_dict: dict = new_lut_classes):
    """
    Returns the new class for a given old class. 
    """
    old_class_name = old_dict.get(old_class_number)
    if old_class_name is None or old_class_name not in new_dict.values():
        return list(new_dict.keys())[list(new_dict.values()).index('other')]
    new_class_number = list(new_dict.keys())[list(new_dict.values()).index(old_class_name)]
    return new_class_number


def transform_to_only_trees_mask(old_mask: np.ndarray, old_dict: dict= lut_classes, new_dict: dict= new_lut_classes):
    """
    Gives the only-tree mask from the old mask that comes from the flair dataset.
    """
    new_mask = np.vectorize(new_class)(old_mask, old_dict, new_dict)
    return new_mask

def convert_to_color(arr_2d: np.ndarray, palette: dict = new_lut_colors) -> np.ndarray:
    rgb_palette = {k: tuple(int(i * 255) for i in hex2color(v)) for k, v in palette.items()}
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in rgb_palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d

def display_samples(images, masks, nb_samples: list, palette=new_lut_colors) -> None:
    indices= random.sample(range(0, len(images)), nb_samples)
    fig, axs = plt.subplots(nrows = nb_samples, ncols = 3, figsize = (20, nb_samples * 6)); fig.subplots_adjust(wspace=0.0, hspace=0.01)
    fig.patch.set_facecolor('black')
    for u, idx in enumerate(indices):
        with rasterio.open(images[idx], 'r') as f:
            im = f.read([1,2,3]).swapaxes(0, 2).swapaxes(0, 1)
        with rasterio.open(masks[idx], 'r') as f:
            mk = f.read([1])
            mk = convert_to_color(transform_to_only_trees_mask(mk[0]), palette=palette)
        axs = axs if isinstance(axs[u], np.ndarray) else [axs]
        ax0 = axs[u][0] ; ax0.imshow(im);ax0.axis('off')
        ax1 = axs[u][1] ; ax1.imshow(mk, interpolation='nearest') ;ax1.axis('off')
        ax2 = axs[u][2] ; ax2.imshow(im); ax2.imshow(mk, interpolation='nearest', alpha=0.25); ax2.axis('off')
        if u == 0:
            ax0.set_title('RVB Image', size=16,fontweight="bold",c='w')
            ax1.set_title('Ground Truth Mask', size=16,fontweight="bold",c='w')
            ax2.set_title('Overlay Image & Mask', size=16,fontweight="bold",c='w')    


def load_data(paths_data, val_percent=0.8, use_metadata=True):
    
    def _gather_data(domains, paths_data: dict, use_metadata: bool, test_set: bool) -> dict:

        #### return data paths
        def get_data_paths (path, filter):
            for path in Path(path).rglob(filter):
                 yield path.resolve().as_posix()        

        #### encode metadata
        def coordenc_opt(coords, enc_size=32) -> np.array:
            d = int(enc_size/2)
            d_i = np.arange(0, d / 2)
            freq = 1 / (10e7 ** (2 * d_i / d))

            x,y = coords[0]/10e7, coords[1]/10e7
            enc = np.zeros(d * 2)
            enc[0:d:2]    = np.sin(x * freq)
            enc[1:d:2]    = np.cos(x * freq)
            enc[d::2]     = np.sin(y * freq)
            enc[d + 1::2] = np.cos(y * freq)
            return list(enc)           

        def norm_alti(alti: int) -> float:
            min_alti = 0
            max_alti = 3164.9099121094
            return [(alti-min_alti) / (max_alti-min_alti)]        

        def format_cam(cam: str) -> np.array:
            return [[1,0] if 'UCE' in cam else [0,1]][0]

        def cyclical_enc_datetime(date: str, time: str) -> list:
            def norm(num: float) -> float:
                return (num-(-1))/(1-(-1))
            year, month, day = date.split('-')
            if year == '2018':   enc_y = [1,0,0,0]
            elif year == '2019': enc_y = [0,1,0,0]
            elif year == '2020': enc_y = [0,0,1,0]
            elif year == '2021': enc_y = [0,0,0,1]    
            sin_month = np.sin(2*np.pi*(int(month)-1/12)) ## months of year
            cos_month = np.cos(2*np.pi*(int(month)-1/12))    
            sin_day = np.sin(2*np.pi*(int(day)/31)) ## max days
            cos_day = np.cos(2*np.pi*(int(day)/31))     
            h,m=time.split('h')
            sec_day = int(h) * 3600 + int(m) * 60
            sin_time = np.sin(2*np.pi*(sec_day/86400)) ## total sec in day
            cos_time = np.cos(2*np.pi*(sec_day/86400))
            return enc_y+[norm(sin_month),norm(cos_month),norm(sin_day),norm(cos_day),norm(sin_time),norm(cos_time)]        

        data = {'IMG':[],'MSK':[],'MTD':[]}   
        for domain in domains: 
            for area in os.listdir(Path(paths_data['path_aerial_'+['train' if test_set == False else 'test'][0]], domain)):
                data['IMG'] += sorted(list(get_data_paths(Path(paths_data['path_aerial_'+['train' if test_set == False else 'test'][0]], domain, area),'IMG*.tif')),
                                      key=lambda x: int(x.split('_')[-1][:-4]),
                                     )
                if test_set == False:
                    data['MSK'] += sorted(list(get_data_paths(Path(paths_data['path_labels_'+['train' if test_set == False else 'test'][0]], domain, area),'MSK*.tif')),
                                          key=lambda x: int(x.split('_')[-1][:-4]),
                                         )        
        if use_metadata == True:

            with open(paths_data['path_metadata_aerial'], 'r') as f:
                metadata_dict = json.load(f)              
            for img in data['IMG']:
                curr_img     = img.split('/')[-1][:-4]
                enc_coords   = coordenc_opt([metadata_dict[curr_img]["patch_centroid_x"], metadata_dict[curr_img]["patch_centroid_y"]])
                enc_alti     = norm_alti(metadata_dict[curr_img]["patch_centroid_z"])
                enc_camera   = format_cam(metadata_dict[curr_img]['camera'])
                enc_temporal = cyclical_enc_datetime(metadata_dict[curr_img]['date'], metadata_dict[curr_img]['time'])
                mtd_enc      = enc_coords+enc_alti+enc_camera+enc_temporal 
                data['MTD'].append(mtd_enc)

        if test_set == False:
            if len(data['IMG']) != len(data['MSK']): 
                print('[WARNING !!] UNMATCHING NUMBER OF IMAGES AND MASKS ! Please check load_data function for debugging.')
            if data['IMG'][0][-10:-4] != data['MSK'][0][-10:-4] or data['IMG'][-1][-10:-4] != data['MSK'][-1][-10:-4]: 
                print('[WARNING !!] UNSORTED IMAGES AND MASKS FOUND ! Please check load_data function for debugging.')                

        data['IMG'] = data['IMG'][:500]
        data['MSK'] = data['MSK'][:500]
        data['MTD'] = data['MTD'][:500]

        return data
    
    
    path_trainval = Path(paths_data['path_aerial_train'])
    trainval_domains = os.listdir(path_trainval)
    shuffle(trainval_domains)
    idx_split = int(len(trainval_domains) * val_percent)
    train_domains, val_domains = trainval_domains[:idx_split], trainval_domains[idx_split:] 
    
    dict_train = _gather_data(train_domains, paths_data, use_metadata=use_metadata, test_set=False)
    dict_val = _gather_data(val_domains, paths_data, use_metadata=use_metadata, test_set=False)
    
    path_test = Path(paths_data['path_aerial_test'])
    test_domains = os.listdir(path_test)
    
    dict_test = _gather_data(test_domains, paths_data, use_metadata=use_metadata, test_set=True)
    
    return dict_train, dict_val, dict_test


@rank_zero_only
def step_loading(paths_data, use_metadata: bool) -> dict:
    print('+'+'-'*29+'+', '   LOADING DATA   ', '+'+'-'*29+'+')
    train, val, test = load_data(paths_data, use_metadata=use_metadata)
    return train, val, test  




def display_predictions(images, predictions, nb_samples: int, palette=new_lut_colors, classes=new_lut_classes) -> None:
    indices= random.sample(range(0, len(predictions)), nb_samples)
    fig, axs = plt.subplots(nrows = nb_samples, ncols = 2, figsize = (17, nb_samples * 8)); fig.subplots_adjust(wspace=0.0, hspace=0.01)
    fig.patch.set_facecolor('black')

    for u, idx in enumerate(indices):
        rgb_image = [i for i in images if predictions[idx].split('_')[-1][:-4] in i][0]
        with rasterio.open(rgb_image, 'r') as f:
            im = f.read([1,2,3]).swapaxes(0, 2).swapaxes(0, 1)
        with rasterio.open(predictions[idx], 'r') as f:
            mk = f.read([1])+1
            f_classes = np.array(list(set(mk.flatten())))
            mk = convert_to_color(mk[0], palette=palette)
        axs = axs if isinstance(axs[u], np.ndarray) else [axs]
        ax0 = axs[u][0] ; ax0.imshow(im);ax0.axis('off')
        ax1 = axs[u][1] ; ax1.imshow(mk, interpolation='nearest', alpha=1); ax1.axis('off')
        if u == 0:
            ax0.set_title('RVB Image', size=16,fontweight="bold",c='w')
            ax1.set_title('Prediction', size=16,fontweight="bold",c='w')
        handles = []
        for val in f_classes:
            handles.append(mpatches.Patch(color=palette[val], label=classes[val]))
        leg = ax1.legend(handles=handles, ncol=1, bbox_to_anchor=(1.4,1.01), fontsize=12, facecolor='k') 
        for txt in leg.get_texts():
          txt.set_color('w')