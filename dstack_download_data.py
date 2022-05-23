import gdown
import os

data_path = './data'
os.makedirs(data_path, exist_ok=True)

def get_file_from_yd(link, output_path):
    import os, sys, json
    import urllib.parse as ul

    base_url = 'https://cloud-api.yandex.net:443/v1/disk/public/resources/download?public_key='
    url = ul.quote_plus(link)
    res = os.popen('wget -qO - {}{}'.format(base_url, url)).read()
    json_res = json.loads(res)
    filename = ul.parse_qs(ul.urlparse(json_res['href']).query)['filename'][0]
    os.system("wget '{}' -P '{}' -O '{}'".format(json_res['href'], output_path, filename))
    # os.system("wget '{}'".format(json_res['href']))

# y_val_id = '1B914UmO6tGvUfaAhDt5mDNAhQrGLk3wi'
# X_val_id = '12B5Rt3eDDoWTsqJReKq_p5zFFcaj4qY1'
# gdown.download(f"https://drive.google.com/uc?id={y_val_id}", os.path.join(data_path, 'y_val.feather'))
# gdown.download(f"https://drive.google.com/uc?id={X_val_id}", os.path.join(data_path, 'X_val.feather'))
y_val_link = 'https://disk.yandex.ru/d/pOkaC9ZAZxysmA'
X_val_link = 'https://disk.yandex.ru/d/lBxxwaKvPt00zQ'
get_file_from_yd(y_val_link, os.path.join(data_path, 'y_val.feather'))
get_file_from_yd(X_val_link, os.path.join(data_path, 'X_val.feather'))
print('val is downloaded')

# y_train_id = '1mLHgHuL5_Frx6CQieFIVZXbzFJYNfczZ'
# X_train_id = '1HfDK5CN3Z6ZvxzfqOMrqrR1z3SLjo9-S'
# gdown.download(f"https://drive.google.com/uc?id={y_train_id}", os.path.join(data_path, 'y_train.feather'))
# gdown.download(f"https://drive.google.com/uc?id={X_train_id}", os.path.join(data_path, 'X_train.feather'))
y_train_link = 'https://disk.yandex.ru/d/m0EjoMSy_TsNOQ'
X_train_link = 'https://disk.yandex.ru/d/kONiRfRvj_NXoQ'
get_file_from_yd(y_train_link, os.path.join(data_path, 'y_train.feather'))
get_file_from_yd(X_train_link, os.path.join(data_path, 'X_train.feather'))
print('train is downloaded')
#
# y_test_id = '1R5s0pTHyv0NTRgfPGVB_8m7u10skRzVz'
# X_test_id = '1-3sgsKXN_SlzYFyHm0E3i09a077f5p8T'
# gdown.download(f"https://drive.google.com/uc?id={y_test_id}", os.path.join(data_path, 'y_test.feather'))
# gdown.download(f"https://drive.google.com/uc?id={X_test_id}", os.path.join(data_path, 'X_test.feather'))
y_test_link = 'https://disk.yandex.ru/d/IHHXhEy4Zt7bcw'
X_test_link = 'https://disk.yandex.ru/d/-QYfzhplNJW2Pg'
get_file_from_yd(y_test_link, os.path.join(data_path, 'y_test.feather'))
get_file_from_yd(X_test_link, os.path.join(data_path, 'X_test.feather'))
print('test is downloaded')