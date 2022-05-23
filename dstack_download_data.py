import gdown
import os

data_path = './data'
os.makedirs(data_path, exist_ok=True)

y_val_id = '1m_cub6Cnu6_IIsEe7EbV2Cy390s46jBT'
X_val_id = '11v3eyriTiuxjP6xlNB9H0MiAiTQpYx1b'

y_train_id = '1b0hoRWJte6Ng5OwnBpqF7FKE1nCTCjCT'
X_train_id = '1uLkqCBmLzmiNOdTJEWzKeU5E4UYRsdZY'

gdown.download(f"https://drive.google.com/uc?id={y_val_id}", os.path.join(data_path, 'y_val.feather'))
gdown.download(f"https://drive.google.com/uc?id={X_val_id}", os.path.join(data_path, 'X_val.feather'))
print('val is downloaded')

gdown.download(f"https://drive.google.com/uc?id={y_train_id}", os.path.join(data_path, 'y_train.feather'))
gdown.download(f"https://drive.google.com/uc?id={X_train_id}", os.path.join(data_path, 'X_train.feather'))
print('train is downloaded')
