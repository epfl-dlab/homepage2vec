import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
# import pandas as pd
from collections import Counter

class TextualExtractor:
    """
    Extract textual features from the html content of a webpage
    """

    xlmr = None

    def __init__(self, device='cpu'):
        if not TextualExtractor.xlmr:
            TextualExtractor.xlmr = SentenceTransformer('paraphrase-xlm-r-multilingual-v1', device=device)
        # self.xlmr = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1', device=device)
        
        # TLD used for one-hot encoding
        self.rep_tld = ['com', 'org', 'net', 'info', 'xyz', 'club', 'biz', 'top', 'edu', 'online', 
                        'pro', 'site', 'vip', 'icu', 'buzz', 'app', 'asia', 'su', 'gov', 'space']
        
        # Metatags used for one-hot encoding
        self.rep_metatags = ['viewport', 'description', 'generator', 'keywords', 'robots', 'twitter:card', 
                             'msapplication-tileimage', 'google-site-verification', 'author', 'twitter:title', 
                             'twitter:description', 'theme-color', 'twitter:image', 'twitter:site', 
                             'format-detection', 'msapplication-tilecolor', 'copyright', 'twitter:data1', 
                             'twitter:label1', 'revisit-after', 'apple-mobile-web-app-capable', 'handheldfriendly', 
                             'language', 'msvalidate.01', 'twitter:url', 'title', 'mobileoptimized', 
                             'twitter:creator', 'skype_toolbar', 'rating']
        
        # number of sentences and links over which we compute the features
        self.k_sentences = 100
        self.k_links = 50

    def get_features(self, url, html):

        features = {}

        # url
        url_feature = embed_url(url, TextualExtractor.xlmr)
        features['f_url'] = url_feature

        # tld 
        tld_feature = embed_tld(url, self.rep_tld)
        features['f_tld'] = tld_feature

        # print(html)
        soup = BeautifulSoup(html, 'lxml')

        # metatags
        metatags_feature = embed_metatags(soup, self.rep_metatags)
        features['f_metatags'] = metatags_feature

        # title
        title_feature = embed_title(soup, TextualExtractor.xlmr)
        features['f_title'] = title_feature

        # description
        description_feature = embed_description(soup, TextualExtractor.xlmr)
        features['f_description'] = description_feature

        # keywords
        keywords_feature = embed_keywords(soup, TextualExtractor.xlmr)
        features['f_keywords'] = keywords_feature

        # links
        links_feature = embed_links(soup, TextualExtractor.xlmr, self.k_links)
        features['f_links_' + str(self.k_links)] = links_feature

        # text
        text_feature = embed_text(soup, TextualExtractor.xlmr, self.k_sentences)
        features['f_text_' + str(self.k_sentences)] = text_feature

        return features

    

def embed_text(soup, transformer, k_sentences):

    sentences = split_in_sentences(soup)[:k_sentences]

    if len(sentences) == 0:
        return None

    # this is needed to avoid some warnings, truncate the sentences
    sentences_trunc = [trunc(s, transformer.tokenizer, transformer.max_seq_length) for s in sentences]

    sentences_emb = transformer.encode(sentences_trunc)

    if sentences_emb.size == 0:
        return None

    text_emb = sentences_emb.mean(axis=0).tolist() # mean of the sentences 

    return text_emb


def embed_description(soup, transformer):

    desc = soup.find('meta', attrs = {'name': ['description', 'Description']})

    if not desc:
        return None

    content = desc.get('content', '')

    if len(content.strip()) == 0:
        return None

    content = clean_field(content)

    # this is needed to avoid some warnings
    desc_trunc = trunc(content, transformer.tokenizer, transformer.max_seq_length)
    desc_emb = transformer.encode(desc_trunc)

    if desc_emb.size == 0:
        return None

    return desc_emb.tolist()


def embed_keywords(soup, transformer):

    kw = soup.find('meta', attrs = {'name': 'keywords'})

    if not kw: 
        return None

    content = kw.get('content', '')

    if len(content.strip()) == 0:
        return None

    # this is needed to avoid some warnings
    kw_trunc = trunc(content, transformer.tokenizer, transformer.max_seq_length)
    kw_emb = transformer.encode(kw_trunc)

    if kw_emb.size == 0:
        return None

    return kw_emb.tolist()


def embed_title(soup, transformer):

    title = soup.find('title')

    if title is None:
        return None

    title = str(title.string)
    title = clean_field(title)

    if len(title) == 0:
        return None

    # this is needed to avoid some warnings
    title_trunc = trunc(title, transformer.tokenizer, transformer.max_seq_length)
    title_emb = transformer.encode(title_trunc)

    if title_emb.size == 0:
        return None

    return title_emb.tolist()


def embed_links(soup, transformer, k_links):

    a_tags = soup.find_all('a', href=True)

    links = [a.get('href', '') for a in a_tags]
    links = [clean_link(link) for link in links]
    links = [link for link in links if len(link) != 0]

    words = [w.lower() for w in ' '.join(links).split(' ') if len(w) != 0]

    if len(words) == 0:
        return None

    most_frequent_words = [w[0] for w in Counter(words).most_common(k_links)]

    # most_frequent_words = pd.Series(words).value_counts()[:k_links].index.values

    # this is needed to avoid some warnings
    words_trunc = [trunc(w, transformer.tokenizer, transformer.max_seq_length) for w in most_frequent_words]
    words_emb = transformer.encode(words_trunc)

    if words_emb.size == 0:
        return None

    links_emb = words_emb.mean(axis=0).tolist()

    return links_emb


def embed_url(url, transformer):

    cleaned_url = clean_url(url)

    # this is needed to avoid some warnings
    url_trunc = [trunc(w, transformer.tokenizer, transformer.max_seq_length) for w in cleaned_url]
    url_emb = transformer.encode(cleaned_url)

    if url_emb.size == 0:
        return None

    return url_emb.mean(axis=0).tolist()



def embed_tld(url, rep_tld):

    tld = url.split('.')[-1]
    rep_onehot = [int(tld.startswith(d)) for d in rep_tld]
    continent_onehot = 7*[0] #TODO

    return rep_onehot + continent_onehot



def embed_metatags(soup, rep_metatags):

    metatags = soup.findAll('meta')
    attr = [m.get('name', None) for m in metatags]
    attr = [a.lower() for a in attr if a != None]

    attr_emb = [int(a in attr) for a in rep_metatags]

    return attr_emb



def split_in_sentences(soup):
    """ From the raw html content of a website, extract the text visible to the user and splits it in sentences """

    sep = soup.get_text('[SEP]').split('[SEP]') # separate text elements with special separators [SEP]
    strip = [s.strip() for s in sep if s != '\n']
    clean = [s for s in strip if len(s) != 0]

    return clean


def clean_url(url):
    url = re.sub(r"www.|http://|https://|-|_", '', url)
    return url.split('.')[:-1]


def clean_field(field):
    field = re.sub(r"\*|\n|\r|\t|\||:|-|–", '', field)
    return field.strip()


def clean_link(link):
    link = re.sub(r"www.|http://|https://|[0-9]+", '', link)
    link = re.sub(r"-|_|=|\?|:", ' ', link)
    link = link.split('/')[1:]
    return ' '.join(link).strip()


def trunc(seq, tok, max_length):
    """ Truncate the output of a tokenizer to a given length, doesn't affect the performances """
    e = tok.encode(seq, truncation=True)
    d = tok.decode(e[1:-1][:max_length-2])
    return d

