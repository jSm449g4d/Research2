import os
import sys
import urllib.request
import zipfile
import importlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.join("./", __file__)))

os.makedirs("datasets", exist_ok=True)
os.makedirs("saves", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
def dl_zip_x(url:str=""):
    urllib.request.urlretrieve(url,os.path.join("./datasets",os.path.basename(url).split("?")[0]))
    with zipfile.ZipFile(os.path.join("./datasets",os.path.basename(url).split("?")[0])) as z:
        z.extractall(os.path.join("./datasets"))
        
# DIV2K
dl_zip_x("http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip")
dl_zip_x("http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip")
importlib.import_module("download_module.div2kRotMake")

# Mars surface image (Curiosity rover) labeled data set
dl_zip_x("https://zenodo.org/record/1049137/files/msl-images.zip?download=1")
importlib.import_module("download_module.mlsChoose")
importlib.import_module("download_module.mlsRotMake")
