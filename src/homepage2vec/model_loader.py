import os
import urllib
import zipfile
from os.path import expanduser
import logging

model_with_visual_url = "https://figshare.com/ndownloader/files/34494836"
model_without_visual_url = "https://figshare.com/ndownloader/files/34494839"


def get_model_path(visual=False):
    if visual:
        model_name = "h2v_1000_100"
        model_url = model_with_visual_url
    else:
        model_name = "h2v_1000_100_text_only"
        model_url = model_without_visual_url
    model_home = os.path.join(expanduser("~"), '.homepage2vec')
    os.makedirs(model_home, exist_ok=True)
    model_folder = os.path.join(model_home, model_name)
    if not os.path.exists(model_folder):
        logging.debug('Downloading model {} in {}'.format(model_name, model_folder))
        filename, headers = urllib.request.urlretrieve(model_url)
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall(model_home)
        logging.debug('Downloading model - Done.')
    else:
        logging.debug('Model {} available in {}'.format(model_name, model_folder))
    return model_home, model_folder
