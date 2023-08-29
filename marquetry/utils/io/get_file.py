import os
import urllib.request

from marquetry.configuration import config


def get_file(url, file_name=None):
    if file_name is None:
        file_name = url[url.rfind("/") + 1:]

    file_path = os.path.join(config.CACHE_DIR, file_name)

    if not os.path.exists(config.CACHE_DIR):
        os.mkdir(config.CACHE_DIR)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)

    try:
        urllib.request.urlretrieve(url, file_path, _show_progress)

    except (Exception, KeyboardInterrupt):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise

    print(" Done")

    return file_path


def _show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    percent = downloaded / total_size * 100
    indicator_num = int(downloaded / total_size * 30)

    percent = percent if percent < 100. else 100.
    indicator_num = indicator_num if indicator_num < 30 else 30

    indicator = "#" * indicator_num + "." * (30 - indicator_num)
    print(bar_template.format(indicator, percent), end="")
