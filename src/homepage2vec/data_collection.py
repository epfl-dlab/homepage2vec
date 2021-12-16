import requests

import time

from selenium import webdriver
from selenium.common.exceptions import WebDriverException

from PIL import Image
import os

import signal
from contextlib import contextmanager


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    """ Set a time limit on the execution of a block"""

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def access_website(url, timeout=10):
    """
    Return the response corresponding to a url, or None if there was a request error
    """

    try:
        # avoid the script to be blocked
        with time_limit(10 * timeout):

            # change user-agent so that we don't look like a bot
            headers = requests.utils.default_headers()
            headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.16; rv:84.0) Gecko/20100101 Firefox/84.0',
            })

            # r_head = requests.head("http://" + url, timeout=timeout, headers=headers)
            if not url.startswith("http://") and not url.startswith("https:"):
                url = "http://"+url
            r_get = requests.get(url, timeout=timeout, headers=headers)

            # head_code = r_head.status_code
            get_code = r_get.status_code
            if r_get.encoding.lower() != 'utf-8':
                r_get.encoding = r_get.apparent_encoding
            text = r_get.text
            content_type = r_get.headers.get('content-type', '?').strip()
            return text, get_code, content_type

    except Exception as e:
        return None


def take_screenshot(url, out_path, in_width=1920, in_height=1080, down_factor=3, quality=85, timeout=30):
    """
    Take a screenshot of a website and save it under the outpath
    """

    out_width = int(in_width / down_factor)
    out_height = int(in_height / down_factor)

    try:

        # driver
        options = webdriver.ChromeOptions()
        options.headless = True

        driver = webdriver.Chrome('chromedriver', options=options)

        driver.set_page_load_timeout(timeout)
        driver.set_window_size(in_width, in_height)

        # access the url
        if not url.startswith("http://") and not url.startswith("https:"):
            url = "http://" + url
        driver.get(url)

        # set the opacity to 0 for elements that might be popup, etc...
        try:
            targets = ["popup", "modal", "cookie"]  # the substrings in the div we want to hide
            target_types = ["class", "id"]  # where the substrings are
            js_script = ""
            for tar in targets:
                for ty in target_types:
                    js_script += "document.styleSheets[0].insertRule('div[" + ty + "*=" + tar + \
                                 "] {opacity: 0 !important}', 0); \n"

            driver.execute_script(js_script)

        # if can't access the css sheet
        except WebDriverException as e:
            pass

        # so that the website's elements are loaded
        time.sleep(2)

        # takes a screenshot (only in png)
        driver.save_screenshot(out_path + '.png')

        # convert the png into a jpeg of lesser dimensions and quality
        img = Image.open(out_path + '.png')
        img = img.convert('RGB')
        img = img.resize((out_width, out_height), Image.ANTIALIAS)
        img.save(out_path + '.jpeg', optimize=True, quality=quality)
        os.remove(out_path + '.png')
        return out_path + '.jpeg'

    except Exception as e:
        print(e)
        return

    finally:
        if 'driver' in locals():
            driver.quit()
