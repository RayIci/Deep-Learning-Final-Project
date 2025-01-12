import os
import gdown
import requests
import tarfile
from tqdm import tqdm

from gan_t2i.utils.logger import error, info, success

def download_file_from_google_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        gdown.download(url, destination, quiet=False)
    except Exception as e:
        error("Failed to download file")
        raise e

def download_file(url, destination, params = None):
    try:
        response = requests.get(url, params=params, stream=True)
    
        if response.status_code == 200:
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
        
            with open(destination, 'wb') as file, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=destination, ascii=True
                ) as bar:
                
                for chunk in response.iter_content(chunk_size=block_size):
                    file.write(chunk)
                    bar.update(len(chunk))
            
            if os.path.getsize(destination) == total_size:
                print("Download completato con successo")
                return
            else:
                print(f"Errore: la dimensione del file scaricato non corrisponde")
                raise Exception("File download failed - file size mismatch")
        else:
            error(f"Failed to download file: status code {response.status_code}")
    except Exception as e:
        error("Failed to download file")
        raise e
        
def extract_tar_gz(file_path, extract_path):
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
    except Exception as e:
        error("Failed to extract file")
        raise e