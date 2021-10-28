# Homepage2Vec - Beta :construction:

---
Language-Agnostic Website Embedding and Classification

## Getting started

### Setup:

Step 1: install the library with pip.
```
pip install homepage2vec
```

Step 2: Install the [Selenium Chrome web driver](https://chromedriver.chromium.org/downloads), and add the folder to the system $PATH variable.

Please note that you need a local copy of Chrome browser (See [Getting started](https://chromedriver.chromium.org/getting-started)).

### Usage:

```python
from homepage2vec.model import WebsiteClassifier

model = WebsiteClassifier()
webpages = model.embed_and_predict(['www.nsf.gov'])

print(webpages[0].scores)
```

```json
{'Arts': 0.018672721460461617, 'Business': 0.01062296237796545,
  'Computers': 0.017558472231030464, 'Games': 1.1537405953276902e-05, 
  'Health': 0.021613001823425293, 'Home': 1.8367260054219514e-05, 
  'Kids_and_Teens': 0.1226280927658081, 'News': 3.7846388295292854e-05, 
  'Recreation': 0.015628756955266, 'Reference': 0.7092769145965576, 
  'Science': 0.9873504042625427, 'Shopping': 0.00010123076208401471, 
  'Society': 0.26334095001220703, 'Sports': 0.0005139540298841894}

```

### Output format:

* url: _Website url_
* is_valid: _True if the request is successful_
* features: _Complete feature vector_
* embedding: _Embedding vector representing the website_
* scores: _Prediction probabilities_

### Customization 

```python
model = WebsiteClassifier(cpu_threads_count=24, dataloader_workers=4)
```

Work in progress...