import logging
from homepage2vec.model import WebsiteClassifier

logging.getLogger().setLevel(logging.DEBUG)

model = WebsiteClassifier(visual=False)

website1 = model.fetch_website('snf.org')

scores, embeddings = model.predict(website1)

print(scores)
print(embeddings)

