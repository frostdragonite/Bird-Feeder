import requests
import os
import sys

def get_image_name(filepath, species):
  path, dirs, files = next(os.walk(filepath))
  file_count = len(files)
  return os.path.join(filepath, species + str(file_count + 1) + '.jpg')

def create_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def download_images(urls, path, species):
    create_folder(path)  # makes sure path exists

    for url in urls:
        image_path = get_image_name(path, species)

        response=requests.get(url,stream=True)

        with open(image_path,'wb') as outfile:
            outfile.write(response.content)