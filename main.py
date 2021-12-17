import logging
from homepage2vec.model import WebsiteClassifier

logging.getLogger().setLevel(logging.DEBUG)

model = WebsiteClassifier()

website = model.fetch_website('epfl.ch')

scores, embeddings = model.predict(website)

print("Classes probabilities:", scores)
print("Embedding:", embeddings)
