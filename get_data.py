import os
import shutil
import zipfile

base_dir = '.'
zip_file= 'dogs-vs-cats'

zip_file_path = os.path.join(base_dir, f'{zip_file}.zip')
if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(zip_file)
    os.remove(zip_file_path)
    print(f'Successfully extracted {zip_file}.')

base_dir = 'dogs-vs-cats'
zip_files = ['test1', 'train']

for zip_file in zip_files:
    zip_file_path = os.path.join(base_dir, f'{zip_file}.zip')
    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        os.remove(zip_file_path)
        print(f'Successfully extracted {zip_file}.')
    else:
        print(f'Error: {zip_file}.zip not found.')
print('Extraction process completed.')


root = r'dogs-vs-cats'
train_dir = r'dogs-vs-cats\train'
test_dir = r'dogs-vs-cats\test1'
res = r'dogs-vs-cats\res'

list_train = os.listdir(train_dir)
list_train

dogs_train_dir = os.path.join(train_dir, 'dogs')
cats_train_dir = os.path.join(train_dir, 'cats')
os.makedirs(dogs_train_dir, exist_ok=True)
os.makedirs(cats_train_dir, exist_ok=True)
for item in list_train:
    item_dir = os.path.join(train_dir, item)
    para = item.split('.')
    if para[0] == 'cat':
        shutil.move(item_dir, cats_train_dir)
    else:
        shutil.move(item_dir, dogs_train_dir)


k = input("Data preparation is completed. Press close to exit") 
