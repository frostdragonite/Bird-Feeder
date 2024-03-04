
## Acknowledgements

 - [MegaDetector CameraTraps (Archived)](https://github.com/microsoft/CameraTraps/blob/main/archive/megadetector.md)
 - [My Dataset that I used to train model](https://drive.google.com/drive/folders/1xhAFhGtBhn50LcQ3_zY3N4ODhk4oT1uu?usp=sharing)
## Installation

You'll need to install Python 3.8 to run this project.

Install all the required libraries using

```bash
  pip install -r requirements.txt
```

You'll also need to get the MegaDetector model from [here.](https://github.com/microsoft/CameraTraps/blob/main/archive/megadetector.md)

Using the [MegaDetector v5a (.pt)](https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt)
## Usage

- Using Flickr to download a dataset of images
- Using `Crops.ipynb` to crop all the images from dataset
- Using `MLModel.ipynb` to train model for classifying birds using cropped dataset
## Running Tests

To run tests, run the following command

```bash
  # Run the Python script
  python RPi/maintest.py
```

This will process all the images in `tester\image` and try to detect a bird inside each image before sending to Telegram.
## Run Locally

To run on the Raspberry Pi, run the following command

```bash
  # Run the Python script
  python RPi/main.py
```

This will host the streaming server as well as detecting any motion from the camera. Sending notification to Telegram whenever it detects a bird.
