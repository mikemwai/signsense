import gdown

url = 'https://drive.google.com/uc?id=1pVNVyrR-xtH9DQx0WYjGCw_YJEmBYcIG'
output = 'vgg_unfrozen.h5'
gdown.download(url, output, quiet=False)