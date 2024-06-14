import os
import gdown
import requests
import tarfile

from gan_t2i.utils.logger import error, info, success

def download_file_from_google_drive(id, destination):
    try:
        url = f'https://drive.google.com/uc?id={id}'
        gdown.download(url, destination, quiet=False)
    except Exception as e:
        error("Failed to download file")
        print(e)


def download_file(url, destination):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
        else:
            error(f"Failed to download file: status code {response.status_code}")
    except Exception as e:
        error("Failed to download file")
        print(e)
        
def extract_tar_gz(file_path, extract_path):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path)