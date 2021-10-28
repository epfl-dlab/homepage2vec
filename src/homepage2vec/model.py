import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
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

    def __init__(self, device=None, cpu_threads_count=1, dataloader_workers=1):
        self.input_dim = 5177
        self.output_dim = 14
        self.classes = ['Arts', 'Business', 'Computers', 'Games', 'Health', 'Home', 'Kids_and_Teens',
                        'News', 'Recreation', 'Reference', 'Science', 'Shopping', 'Society', 'Sports']

        self.model_path = self.get_model_path()

        self.temporary_dir = tempfile.gettempdir() + "/homepage2vec/"
        self.dataloader_workers = dataloader_workers
        # print(self.temporary_dir)

        os.makedirs(self.temporary_dir + "/screenshots", exist_ok=True)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # self.temporary_dir="/tmp/screenshots/"
        files = glob.glob(self.temporary_dir + "/screenshots/*")
        for f in files:
            os.remove(f)

        self.device = device
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

    def get_model_path(self):
        MODEL_NAME = "final_1000_100_posw_heldout"
        model_home = os.path.join(expanduser("~"), '.homepage2vec')
        os.makedirs(model_home, exist_ok=True)
        model_folder = os.path.join(model_home, MODEL_NAME)
        if not os.path.exists(model_folder):
            filename, headers = urllib.request.urlretrieve(
                "https://figshare.com/ndownloader/files/31247047?private_link=e664b0204a98a94cfe3c")
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(model_home)
        return model_folder

    def compute_features(self, webpages):
        """
        Given list of webpages, computer their textual and visual features and store them in instances attributes.
        Return a list with valid webpages, whose attributes have been updated.
        """

        te = TextualExtractor(self.device)
        ve = VisualExtractor(self.device)

        # possible parallelism here for speed-up
        web_ix = 0
        for w in webpages:

            w.is_valid = False  # invalid by default
            response = access_website(w.url)

            if response is not None:
                html, head_code, get_code, content_type = response

                if self.is_valid(get_code, content_type):
                    w.is_valid = True

                    # take screenshot
                    out_path = self.temporary_dir + "/screenshots/" + str(w.uid)
                    take_screenshot(w.url, out_path)

                    # compute textual features
                    w.features = te.get_features(w.url, html)

            web_ix += 1

        # compute visual features for all webpages, process screenshots in batches
        visual_features = ve.get_features(self.temporary_dir, self.dataloader_workers)

        valid_webpages = []

        # retrieve visual features of each webpage
        for w in webpages:
            if w.is_valid:
                w.features['f_visual'] = visual_features[w.uid]
                valid_webpages.append(w)

        return valid_webpages

    def embed_and_predict(self, webpages):
        """
        Given list of valid webpages with features, compute their classes scores and embedding
        """
        # print(self)

        webpages = [Webpage(w) for w in webpages]

        valid_webpages = self.compute_features(webpages)

        features_matrix = np.zeros((len(valid_webpages), self.input_dim))
        for i in range(len(valid_webpages)):
            features_matrix[i, :] = self.concatenate_features(valid_webpages[i])

        scores, embeddings = self.get_scores(torch.FloatTensor(features_matrix))
        for i in range(len(valid_webpages)):
            valid_webpages[i].embedding = embeddings[i].tolist()
            valid_webpages[i].scores = dict(zip(self.classes, torch.sigmoid(scores[i]).tolist()))

        return webpages

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
        self.is_valid = None
        self.features = None
        self.embedding = None
        self.scores = None

    def __repr__(self):
        return json.dumps(self.__dict__)
