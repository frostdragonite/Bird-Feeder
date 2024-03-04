import xml.etree.ElementTree as ET
import requests
import os
import sys

path = os.path.join('data', 'bird')
if not os.path.isdir(path):
    os.makedirs(path)
        
tree = ET.parse('flickr/urls.xml')
root = tree.getroot()
for i in range(len(root)):
    for photo in root[i]:
        url = photo.get('url_m')
        image_name = url.split("/")[-1]
        image_path = os.path.join(path, image_name)
        if not os.path.isfile(image_path):  # ignore if already downloaded
            response=requests.get(url,stream=True)

            with open(image_path,'wb') as outfile:
                outfile.write(response.content)
                
    print("Page " + str(i) + " Done")