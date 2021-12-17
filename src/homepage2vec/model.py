import logging

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from homepage2vec.model_loader import get_model_path
from homepage2vec.textual_extractor import TextualExtractor
from homepage2vec.visual_extractor import VisualExtractor
from homepage2vec.data_collection import access_website, take_screenshot
import uuid
import tempfile
import os
from os.path import expanduser
import glob
import json
import urllib.request
import zipfile


class WebsiteClassifier:
    """
    Pretrained Homepage2vec model
    """

    def __init__(self, visual=False, device=None, cpu_threads_count=1, dataloader_workers=1):
        self.input_dim = 5177 if visual else 4665
        self.output_dim = 14
        self.classes = ['Arts', 'Business', 'Computers', 'Games', 'Health', 'Home', 'Kids_and_Teens',
                        'News', 'Recreation', 'Reference', 'Science', 'Shopping', 'Society', 'Sports']

        self.with_visual = visual

        self.model_home_path, self.model_path = get_model_path(visual)

        self.temporary_dir = tempfile.gettempdir() + "/homepage2vec/"
        os.makedirs(self.temporary_dir + "/screenshots", exist_ok=True)
        # clean screen shorts
        files = glob.glob(self.temporary_dir + "/screenshots/*")
        for f in files:
            os.remove(f)

        self.device = device
        self.dataloader_workers = dataloader_workers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if not device:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'
                torch.set_num_threads(cpu_threads_count)

        # load pretrained model
        model_tensor = torch.load(self.model_path + "/model.pt", map_location=torch.device(self.device))
        self.model = SimpleClassifier(self.input_dim, self.output_dim)
        self.model.load_state_dict(model_tensor)

        # features used in training
        self.features_order = []
        self.features_dim = {}
        logging.debug("Loading features from {}".format(self.model_path + '/features.txt'))
        with open(self.model_path + '/features.txt', 'r') as file:
            for f in file:
                name = f.split(' ')[0]
                dim = int(f.split(' ')[1][:-1])
                self.features_order.append(name)
                self.features_dim[name] = dim

    def get_scores(self, x):
        with torch.no_grad():
            self.model.eval()
            return self.model.forward(x)

    def fetch_website(self, url):
        logging.debug("Fetching website: {}".format(url))
        response = access_website(url)
        w = Webpage(url)
        if response is not None:
            html, get_code, content_type = response
            w.http_code = get_code
            if self.is_valid(get_code, content_type):
                w.is_valid = True
                w.html = html
        if self.with_visual:
            logging.debug("Generating screenshot: {}".format(url))
            out_path = self.temporary_dir + "/screenshots/" + str(w.uid)
            w.screenshot_path = take_screenshot(w.url, out_path)
            logging.debug("Screenshot for {} ready in {}".format(url, w.screenshot_path))
        return w

    def get_features(self, url, html, screenshot_path):
        te = TextualExtractor(self.device)
        features = te.get_features(url, html)
        if self.with_visual:
            ve = VisualExtractor(self.device)
            visual_features = ve.get_features(screenshot_path)
            features['f_visual'] = visual_features
        return features

    def predict(self, website):
        website.features = self.get_features(website.url, website.html, website.screenshot_path)
        all_features = self.concatenate_features(website)
        input_features = torch.FloatTensor(all_features)
        scores, embeddings = self.get_scores(input_features)
        return dict(zip(self.classes, torch.sigmoid(scores).tolist())), embeddings.tolist()


    def concatenate_features(self, w):
        """
        Concatenate the features attributes of webpage instance, with respect to the features order in h2v
        """

        v = np.zeros(self.input_dim)

        ix = 0

        for f_name in self.features_order:
            f_dim = self.features_dim[f_name]
            f_value = w.features[f_name]
            if f_value is None:
                f_value = f_dim * [0]  # if no feature, replace with zeros
            v[ix:ix + f_dim] = f_value
            ix += f_dim

        return v

    def is_valid(self, get_code, content_type):
        valid_get_code = get_code == 200
        valid_content_type = content_type.startswith('text/html')
        return valid_get_code and valid_content_type


class SimpleClassifier(nn.Module):
    """
    Model architecture of Homepage2vec
    """

    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(SimpleClassifier, self).__init__()

        self.layer1 = torch.nn.Linear(input_dim, 1000)
        self.layer2 = torch.nn.Linear(1000, 100)
        self.fc = torch.nn.Linear(100, output_dim)

        self.drop = torch.nn.Dropout(dropout)  # dropout of 0.5 before each layer

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(self.drop(x))

        emb = self.layer2(x)
        x = F.relu(self.drop(emb))

        x = self.fc(x)

        return x, emb


class Webpage:
    """
    Shell for a webpage query
    """

    def __init__(self, url):
        self.url = url
        self.uid = uuid.uuid4().hex
        self.is_valid = False
        self.http_code = False
        self.html = None
        self.screenshot_path = None
        self.features = None
        self.embedding = None
        self.scores = None

    def __repr__(self):
        return json.dumps(self.__dict__)
