import os
import gdown

DATA_PATH = './data'
os.makedirs(DATA_PATH, exist_ok=True)
DOWNLOAD_METHOD = 'google'  # google or yandex
assert DOWNLOAD_METHOD in ['google', 'yandex']

if DOWNLOAD_METHOD == 'yandex':
    import json
    import urllib.parse as ul
    def get_file_from_yd(link, output_path):
        base_url = 'https://cloud-api.yandex.net:443/v1/disk/public/resources/download?public_key='
        url = ul.quote_plus(link)
        res = os.popen('wget -qO - {}{}'.format(base_url, url)).read()
        json_res = json.loads(res)
        filename = ul.parse_qs(ul.urlparse(json_res['href']).query)['filename'][0]
        os.system("wget '{}' -P '{}' -O '{}'".format(json_res['href'], output_path, filename))
        # os.system("wget '{}'".format(json_res['href']))

if DOWNLOAD_METHOD == 'google':
    Y_VAL_ID = '1B914UmO6tGvUfaAhDt5mDNAhQrGLk3wi'
    X_VAL_ID = '12B5Rt3eDDoWTsqJReKq_p5zFFcaj4qY1'
    gdown.download(f"https://drive.google.com/uc?id={Y_VAL_ID}", os.path.join(DATA_PATH, 'y_val.feather'))
    gdown.download(f"https://drive.google.com/uc?id={X_VAL_ID}", os.path.join(DATA_PATH, 'X_val.feather'))
else:
    Y_VAL_LINK = 'https://disk.yandex.ru/d/pOkaC9ZAZxysmA'
    X_VAL_LINK = 'https://disk.yandex.ru/d/lBxxwaKvPt00zQ'
    get_file_from_yd(Y_VAL_LINK, os.path.join(DATA_PATH, 'y_val.feather'))
    get_file_from_yd(X_VAL_LINK, os.path.join(DATA_PATH, 'X_val.feather'))
print('val is downloaded')

if DOWNLOAD_METHOD == 'google':
    Y_TRAIN_ID = '1mLHgHuL5_Frx6CQieFIVZXbzFJYNfczZ'
    X_TRAIN_ID = '1HfDK5CN3Z6ZvxzfqOMrqrR1z3SLjo9-S'
    gdown.download(f"https://drive.google.com/uc?id={Y_TRAIN_ID}", os.path.join(DATA_PATH, 'y_train.feather'))
    gdown.download(f"https://drive.google.com/uc?id={X_TRAIN_ID}", os.path.join(DATA_PATH, 'X_train.feather'))
else:
    Y_TRAIN_LINK = 'https://disk.yandex.ru/d/m0EjoMSy_TsNOQ'
    X_TRAIN_LINK = 'https://disk.yandex.ru/d/kONiRfRvj_NXoQ'
    get_file_from_yd(Y_TRAIN_LINK, os.path.join(DATA_PATH, 'y_train.feather'))
    get_file_from_yd(X_TRAIN_LINK, os.path.join(DATA_PATH, 'X_train.feather'))
print('train is downloaded')

if DOWNLOAD_METHOD == 'google':
    Y_TEST_ID = '1R5s0pTHyv0NTRgfPGVB_8m7u10skRzVz'
    X_TEST_ID = '1-3sgsKXN_SlzYFyHm0E3i09a077f5p8T'
    gdown.download(f"https://drive.google.com/uc?id={Y_TEST_ID}", os.path.join(DATA_PATH, 'y_test.feather'))
    gdown.download(f"https://drive.google.com/uc?id={X_TEST_ID}", os.path.join(DATA_PATH, 'X_test.feather'))
else:
    Y_TEST_LINK = 'https://disk.yandex.ru/d/IHHXhEy4Zt7bcw'
    X_TEST_LINK = 'https://disk.yandex.ru/d/-QYfzhplNJW2Pg'
    get_file_from_yd(Y_TEST_LINK, os.path.join(DATA_PATH, 'y_test.feather'))
    get_file_from_yd(X_TEST_LINK, os.path.join(DATA_PATH, 'X_test.feather'))
print('test is downloaded')
