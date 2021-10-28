# Homepage2Vec - Beta :construction:

---
Language-Agnostic Website Embedding and Classification

## Getting started

### Setup:
```
pip install remotipy
```

You need Selenium Chrome web driver in the $PATH: https://chromedriver.chromium.org/downloads

### Usage:

```python
from homepage2vec.model import WebsiteClassifier

model = WebsiteClassifier()
webpages = model.embed_and_predict(['www.nsf.gov'])

print(webpages[0].scores)
```