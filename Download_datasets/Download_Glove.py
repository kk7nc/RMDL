from __future__ import print_function

import os, sys, tarfile
import numpy as np
import zipfile

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib

print(sys.version_info)

# image shape


# path to the directory with the data
DATA_DIR = '.\Glove'

# url of the binary data
DATA_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'


# path to the binary train file with image data


def download_and_extract():
    """
    Download and extract the GloVe
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    print(filepath)

    path = os.path.abspath(dest_directory)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                          float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)


        zip_ref = zipfile.ZipFile(filepath, 'r')
        zip_ref.extractall(DATA_DIR)
        zip_ref.close()
    return path
