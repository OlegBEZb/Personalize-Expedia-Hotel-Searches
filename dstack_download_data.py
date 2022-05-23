import gdown
import os

data_path = './data'
os.makedirs(data_path, exist_ok=True)

y_val_id = '1m_cub6Cnu6_IIsEe7EbV2Cy390s46jBT'
X_val_id = '11v3eyriTiuxjP6xlNB9H0MiAiTQpYx1b'
gdown.download(f"https://drive.google.com/uc?id={y_val_id}", os.path.join(data_path, 'y_val.feather'))
gdown.download(f"https://drive.google.com/uc?id={X_val_id}", os.path.join(data_path, 'X_val.feather'))
print('val is downloaded')

y_train_id = '1b0hoRWJte6Ng5OwnBpqF7FKE1nCTCjCT'
X_train_id = '1zuKDqZek3eck32DMLajjzKdRPd5qzMn3'
gdown.download(f"https://drive.google.com/uc?id={y_train_id}", os.path.join(data_path, 'y_train.feather'))
gdown.download(f"https://drive.google.com/uc?id={X_train_id}", os.path.join(data_path, 'X_train.feather'))
print('train is downloaded')

y_test_id = '1CTpoyFRONgLPUFIqKklTI6UOLol_zAAP'
X_test_id = '1VLto2yZqi-vL6ze4xhknDBvRppNeaJvO'
gdown.download(f"https://drive.google.com/uc?id={y_test_id}", os.path.join(data_path, 'y_test.feather'))
gdown.download(f"https://drive.google.com/uc?id={X_test_id}", os.path.join(data_path, 'X_test.feather'))
print('test is downloaded')