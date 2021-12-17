# Homepage2Vec - Beta :construction:

---
Language-Agnostic Website Embedding and Classification

## Getting started

### Setup:

Step 1: install the library with pip.
```
pip install homepage2vec
```

[//]: # ()
[//]: # ([Optional] Step 2: Install the [Selenium Chrome web driver]&#40;https://chromedriver.chromium.org/downloads&#41;, and add the folder to the system $PATH variable.)

[//]: # ()
[//]: # (Please note that you need a local copy of Chrome browser &#40;See [Getting started]&#40;https://chromedriver.chromium.org/getting-started&#41;&#41;.)

### Usage:

```python
import logging
from homepage2vec.model import WebsiteClassifier

logging.getLogger().setLevel(logging.DEBUG)

model = WebsiteClassifier()

website = model.fetch_website('epfl.ch')

scores, embeddings = model.predict(website)

print("Classes probabilities:", scores)
print("Embedding:", embeddings)
```
Result:
```
Classes probabilities: {'Arts': 0.3674524128437042, 'Business': 0.0720655769109726,
 'Computers': 0.03488553315401077, 'Games': 7.529282356699696e-06, 
 'Health': 0.02021787129342556, 'Home': 0.0005890956381335855, 
 'Kids_and_Teens': 0.3113572597503662, 'News': 0.0079914266243577, 
 'Recreation': 0.00835705827921629, 'Reference': 0.931416392326355, 
 'Science': 0.959597110748291, 'Shopping': 0.0010162043618038297, 
 'Society': 0.23374591767787933, 'Sports': 0.00014659571752417833}
 
Embedding: [-4.596550941467285, 1.0690114498138428, 2.1633379459381104,
 0.1665923148393631, -4.605356216430664, -2.894961357116699, 0.5615459084510803, 
 1.6420538425445557, -1.918184757232666, 1.227172613143921, 0.4358430504798889, 
 ...]
```
