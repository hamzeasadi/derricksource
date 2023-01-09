import requests
import os
import conf as cfg
import argparse

parser = argparse.ArgumentParser(prog='download.py', description='this scripets take file id of google drive file and save it in specific location')
parser.add_argument('--fileid', '-fi', type=str, required=True, default='None')
parser.add_argument('--savepath', '-sp', type=str, required=True, default='./')
parser.add_argument('--filename', '-fn', type=str, required=True, default='data.zip')

args = parser.parse_args()

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id , 'confirm': 1 }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)




if __name__ == "__main__":
    # file_id = '1c84_CtRGK8ifv-7c9MI_xYVruYAnYX5p'
    file_id = args.fileid
    destination = os.path.join(cfg.paths[args.savepath], args.filename)
    download_file_from_google_drive(file_id, destination)
