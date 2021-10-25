import torch

from homepage2vec.model import WebsiteClassifier, Webpage

model_path = "/Users/piccardi/repos/multilang-web-embedding/training/results/final_1000_100_posw_heldout"

model = WebsiteClassifier(model_path=model_path)

w1 = Webpage('www.epfl.ch')

webpages = model.embed_and_predict([w1])
