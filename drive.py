# download_from_drive.py
import gdown

url = 'https://drive.google.com/file/d/1pVNVyrR-xtH9DQx0WYjGCw_YJEmBYcIG/view?usp=sharing'
output = 'vgg_unfrozen.h5'
gdown.download(url, output, quiet=False)
