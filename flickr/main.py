from flickr import get_urls
from flickr_load import download_images
import os
import time

all_species = ['cat', 'dog', 'rodent', 'snake', 'squirrel', 'raccoon', 'rabbit', 'fox', 'deer', 'bear']
images_per_species = 25

def download():
    for species in all_species:

        print('Getting urls for', species)
        urls = get_urls(species, images_per_species)

        print('Downlaing images for', species)
        path = os.path.join('flickr', 'images')

        download_images(urls, path, species)

if __name__=='__main__':

    start_time = time.time()

    download()

    print('Took', round(time.time() - start_time, 2), 'seconds')