import gdown
import os

data_path = './data'
os.makedirs(data_path, exist_ok=True)

y_val_id = '1B914UmO6tGvUfaAhDt5mDNAhQrGLk3wi'
X_val_id = '12B5Rt3eDDoWTsqJReKq_p5zFFcaj4qY1'
gdown.download(f"https://drive.google.com/uc?id={y_val_id}", os.path.join(data_path, 'y_val.feather'))
gdown.download(f"https://drive.google.com/uc?id={X_val_id}", os.path.join(data_path, 'X_val.feather'))
print('val is downloaded')

y_train_id = '1mLHgHuL5_Frx6CQieFIVZXbzFJYNfczZ'
X_train_id = '1HfDK5CN3Z6ZvxzfqOMrqrR1z3SLjo9-S'
gdown.download(f"https://drive.google.com/uc?id={y_train_id}", os.path.join(data_path, 'y_train.feather'))
gdown.download(f"https://drive.google.com/uc?id={X_train_id}", os.path.join(data_path, 'X_train.feather'))
print('train is downloaded')

y_test_id = '1R5s0pTHyv0NTRgfPGVB_8m7u10skRzVz'
X_test_id = '1-3sgsKXN_SlzYFyHm0E3i09a077f5p8T'
gdown.download(f"https://drive.google.com/uc?id={y_test_id}", os.path.join(data_path, 'y_test.feather'))
gdown.download(f"https://drive.google.com/uc?id={X_test_id}", os.path.join(data_path, 'X_test.feather'))
print('test is downloaded')