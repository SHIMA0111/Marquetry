import os
import urllib.request

from marquetry.configuration import config


def get_file(url, file_name=None):
    """Download a file from a URL and save it to a local cache directory.

        This function downloads a file from the specified URL and saves it to a local cache directory.
        If a local copy of the file with the same name already exists in the cache directory, it is not re-downloaded.

        Args:
            url (str): The URL of the file to be downloaded.
            file_name (str): The name to be used for the downloaded file.
                If None, the last part of the URL (after the last '/') is used as the file name.

        Returns:
            str: The path to the downloaded file in the local cache directory.
    """

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
