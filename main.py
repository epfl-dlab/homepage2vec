import torch

from homepage2vec.model import WebsiteClassifier, Webpage
# from homepage2vec.encoder import compute_features, embed_and_predict

model_path = "/Users/piccardi/repos/multilang-web-embedding/training/results/final_1000_100_posw_heldout"

model = WebsiteClassifier(model_path=model_path)

w1 = Webpage('www.epfl.ch')

valid_webpages = model.compute_features([w1])
