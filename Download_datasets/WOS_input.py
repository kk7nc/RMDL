from __future__ import print_function
import os, sys, tarfile
import time
import numpy as np

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib

print(sys.version_info)

# image shape


# path to the directory with the data
DATA_DIR = '.\data_WOS'

# url of the binary data
DATA_URL = 'http://kowsari.net/WebOfScience.tar.gz'


# path to the binary train file with image data


def download_and_extract():
    """
    Download and extract the WOS dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    print(filepath)


    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()-1
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write("\rDownloading Web Of Science Datasets ...%d%%, %d MB, %d KB/s, %d seconds passed" %
                             (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('\n Downloaded', filename)

        tarfile.open(filepath, 'r').extractall(dest_directory)

