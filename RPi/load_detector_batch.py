r"""
Module to run an animal detection model on lots of images, writing the results
to a file in the same format produced by our batch API:

https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing

This enables the results to be used in our post-processing pipeline; see
api/batch_processing/postprocessing/postprocess_batch_results.py .

This script can save results to checkpoints intermittently, in case disaster
strikes. To enable this, set --checkpoint_frequency to n > 0, and results 
will be saved as a checkpoint every n images. Checkpoints will be written 
to a file in the same directory as the output_file, and after all images
are processed and final results file written to output_file, the temporary
checkpoint file will be deleted. If you want to resume from a checkpoint, set
the checkpoint file's path using --resume_from_checkpoint.

The `threshold` you can provide as an argument is the confidence threshold above
which detections will be included in the output file.

Has preliminary multiprocessing support for CPUs only; if a GPU is available, it will
use the GPU instead of CPUs, and the --ncores option will be ignored.  Checkpointing
is not supported when using multiprocessing.

Does not have a command-line option to bind the process to a particular GPU, but you can 
prepend with "CUDA_VISIBLE_DEVICES=0 ", for example, to bind to GPU 0, e.g.:

CUDA_VISIBLE_DEVICES=0 python detection/run_detector_batch.py ~/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb ~/data/test-small ~/tmp/mdv4test.json --output_relative_filenames --recursive

"""

#%% Constants, imports, environment

import argparse
import json
import os
import sys
import time
import copy
import shutil
import warnings
import itertools

from datetime import datetime
from functools import partial

import humanfriendly
from tqdm import tqdm

# from multiprocessing.pool import ThreadPool as workerpool
import multiprocessing
from threading import Thread
from multiprocessing import Process
from multiprocessing.pool import Pool as workerpool

# Number of images to pre-fetch
max_queue_size = 10
use_threads_for_queue = False
verbose = False

# Useful hack to force CPU inference.
#
# Need to do this before any PT/TF imports, which happen when we import
# from run_detector.
force_cpu = False
if force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import detection.run_detector as run_detector
from detection.run_detector import ImagePathUtils,\
    is_gpu_available,\
    load_detector,\
    get_detector_version_from_filename,\
    get_detector_metadata_from_version_string

import visualization.visualization_utils as viz_utils

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)


#%% Support functions for multiprocessing

def producer_func(q,image_files):
    """ 
    Producer function; only used when using the (optional) image queue.
    
    Reads up to N images from disk and puts them on the blocking queue for processing.
    """
    
    if verbose:
        print('Producer starting'); sys.stdout.flush()
        
    for im_file in image_files:
    
        try:
            if verbose:
                print('Loading image {}'.format(im_file)); sys.stdout.flush()
            image = viz_utils.load_image(im_file)
        except Exception as e:
            print('Producer process: image {} cannot be loaded. Exception: {}'.format(im_file, e))
            raise
        
        if verbose:
            print('Queueing image {}'.format(im_file)); sys.stdout.flush()
        q.put([im_file,image])                    
    
    q.put(None)
        
    print('Finished image loading'); sys.stdout.flush()
    
    
def consumer_func(q,return_queue,model_file,confidence_threshold,image_size=None):
    """ 
    Consumer function; only used when using the (optional) image queue.
    
    Pulls images from a blocking queue and processes them.
    """
    
    if verbose:
        print('Consumer starting'); sys.stdout.flush()

    start_time = time.time()
    detector = load_detector(model_file)
    elapsed = time.time() - start_time
    print('Loaded model (before queueing) in {}'.format(humanfriendly.format_timespan(elapsed)))
    sys.stdout.flush()
        
    results = []
    
    n_images_processed = 0
    
    while True:
        r = q.get()
        if r is None:
            q.task_done()
            return_queue.put(results)
            return
        n_images_processed += 1
        im_file = r[0]
        image = r[1]
        if verbose or ((n_images_processed % 10) == 0):
            elapsed = time.time() - start_time
            images_per_second = n_images_processed / elapsed
            print('De-queued image {} ({:.2f}/s) ({})'.format(n_images_processed,
                                                          images_per_second,
                                                          im_file));
            sys.stdout.flush()
        results.append(process_image(im_file=im_file,detector=detector,
                                     confidence_threshold=confidence_threshold,
                                     image=image,quiet=True,image_size=image_size))
        if verbose:
            print('Processed image {}'.format(im_file)); sys.stdout.flush()
        q.task_done()
            

def run_detector_with_image_queue(image_files,model_file,confidence_threshold,
                                  quiet=False,image_size=None):
    """
    Driver function for the (optional) multiprocessing-based image queue; only used 
    when --use_image_queue is specified.  Starts a reader process to read images from disk, but 
    processes images in the  process from which this function is called (i.e., does not currently
    spawn a separate consumer process).
    """
    
    q = multiprocessing.JoinableQueue(max_queue_size)
    return_queue = multiprocessing.Queue(1)
    
    if use_threads_for_queue:
        producer = Thread(target=producer_func,args=(q,image_files,))
    else:
        producer = Process(target=producer_func,args=(q,image_files,))
    producer.daemon = False
    producer.start()
 
    # TODO
    #
    # The queue system is a little more elegant if we start one thread for reading and one
    # for processing, and this works fine on Windows, but because we import TF at module load,
    # CUDA will only work in the main process, so currently the consumer function runs here.
    #
    # To enable proper multi-GPU support, we may need to move the TF import to a separate module
    # that isn't loaded until very close to where inference actually happens.
    run_separate_consumer_process = False

    if run_separate_consumer_process:
        if use_threads_for_queue:
            consumer = Thread(target=consumer_func,args=(q,return_queue,model_file,
                                                         confidence_threshold,image_size,))
        else:
            consumer = Process(target=consumer_func,args=(q,return_queue,model_file,
                                                          confidence_threshold,image_size,))
        consumer.daemon = True
        consumer.start()
    else:
        consumer_func(q,return_queue,model_file,confidence_threshold,image_size)

    producer.join()
    print('Producer finished')
   
    if run_separate_consumer_process:
        consumer.join()
        print('Consumer finished')
    
    q.join()
    print('Queue joined')

    results = return_queue.get()
    
    return results


#%% Other support funtions

def chunks_by_number_of_chunks(ls, n):
    """
    Splits a list into n even chunks.

    Args
    - ls: list
    - n: int, # of chunks
    """
    for i in range(0, n):
        yield ls[i::n]


#%% Image processing functions

def process_images(im_files, detector, confidence_threshold, use_image_queue=False, 
                   quiet=False, image_size=None):
    """
    Runs MegaDetector over a list of image files.

    Args
    - im_files: list of str, paths to image files
    - detector: loaded model or str (path to .pb/.pt model file)
    - confidence_threshold: float, only detections above this threshold are returned

    Returns
    - results: list of dict, each dict represents detections on one image
        see the 'images' key in https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    
    if isinstance(detector, str):
        start_time = time.time()
        detector = load_detector(detector)
        elapsed = time.time() - start_time
        print('Loaded model (batch level) in {}'.format(humanfriendly.format_timespan(elapsed)))

    if use_image_queue:
        run_detector_with_image_queue(im_files, detector, confidence_threshold, 
                                      quiet=quiet, image_size=image_size)
    else:
        results = []
        for im_file in im_files:
            results.append(process_image(im_file, detector, confidence_threshold,
                                         quiet=quiet, image_size=image_size))
        return results
    

def process_image(im_file, detector, confidence_threshold, image=None, 
                  quiet=False, image_size=None):
    """
    Runs MegaDetector over a single image file.

    Args
    - im_file: str, path to image file
    - detector: loaded model
    - confidence_threshold: float, only detections above this threshold are returned
    - image: previously-loaded image, if available

    Returns:
    - result: dict representing detections on one image
        see the 'images' key in https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
    """
    
    if not quiet:
        print('Processing image {}'.format(im_file))
    
    if image is None:
        try:
            image = viz_utils.load_image(im_file)
        except Exception as e:
            if not quiet:
                print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': run_detector.FAILURE_IMAGE_OPEN
            }
            return result

    try:
        result = detector.generate_detections_one_image(
            image, im_file, detection_threshold=confidence_threshold, image_size=image_size)
    except Exception as e:
        if not quiet:
            print('Image {} cannot be processed. Exception: {}'.format(im_file, e))
        result = {
            'file': im_file,
            'failure': run_detector.FAILURE_INFER
        }
        return result

    return result


#%% Main function

def detector_batch_loader(model_file, n_cores=1):
    """
    Args
    - model_file: str, path to .pb model file
    - n_cores: int, # of CPU cores to use

    Returns
    - results: the model
    """
    
    if n_cores is None:
        n_cores = 1
        
    print('GPU available: {}'.format(is_gpu_available(model_file)))
    
    if n_cores > 1 and is_gpu_available(model_file):
        print('Warning: multiple cores requested, but a GPU is available; parallelization across ' + \
              'GPUs is not currently supported, defaulting to one GPU')
        n_cores = 1

    # Load the detector
    start_time = time.time()
    detector = load_detector(model_file)
    elapsed = time.time() - start_time
    print('Loaded model in {}'.format(humanfriendly.format_timespan(elapsed)))
        
    return detector

def write_results_to_file(results, output_file, relative_path_base=None, 
                          detector_file=None, info=None, include_max_conf=False):
    """
    Writes list of detection results to JSON output file. Format matches:

    https://github.com/ecologize/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format

    Args
    - results: list of dict, each dict represents detections on one image
    - output_file: str, path to JSON output file, should end in '.json'
    - relative_path_base: str, path to a directory as the base for relative paths
    """
    
    if relative_path_base is not None:
        results_relative = []
        for r in results:
            r_relative = copy.copy(r)
            r_relative['file'] = os.path.relpath(r_relative['file'], start=relative_path_base)
            results_relative.append(r_relative)
        results = results_relative

    # The typical case: we need to build the 'info' struct
    if info is None:
        
        info = { 
            'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.3' 
        }
        
        if detector_file is not None:
            detector_filename = os.path.basename(detector_file)
            detector_version = get_detector_version_from_filename(detector_filename)
            detector_metadata = get_detector_metadata_from_version_string(detector_version)
            info['detector'] = detector_filename  
            info['detector_metadata'] = detector_metadata
        else:
            info['detector'] = 'unknown'
            info['detector_metadata'] = get_detector_metadata_from_version_string('unknown')
        
    # If the caller supplied the entire "info" struct
    else:
        
        if detector_file is not None:
            
            print('Warning (write_results_to_file): info struct and detector file ' + \
                  'supplied, ignoring detector file')

    # The 'max_detection_conf' field used to be included by default, and it caused all kinds
    # of headaches, so it's no longer included unless the user explicitly requests it.
    if not include_max_conf:
        for im in results:
            if 'max_detection_conf' in im:
                del im['max_detection_conf']
            
    final_output = {
        'images': results,
        'detection_categories': run_detector.DEFAULT_DETECTOR_LABEL_MAP,
        'info': info
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1)
    print('Output file saved at {}'.format(output_file))


#%% Interactive driver

if False:
    
    pass

    #%%
    
    checkpoint_path = None
    model_file = r'G:\temp\models\md_v4.1.0.pb'
    confidence_threshold = 0.1
    checkpoint_frequency = -1
    results = None
    ncores = 1
    use_image_queue = False
    quiet = False
    image_dir = r'G:\temp\demo_images\ssmini'
    image_size = None
    image_file_names = ImagePathUtils.find_images(image_dir, recursive=False)    
    
    start_time = time.time()
    
    results = load_and_run_detector_batch(model_file=model_file,
                                          image_file_names=image_file_names,
                                          checkpoint_path=checkpoint_path,
                                          confidence_threshold=confidence_threshold,
                                          checkpoint_frequency=checkpoint_frequency,
                                          results=results,
                                          n_cores=ncores,
                                          use_image_queue=use_image_queue,
                                          quiet=quiet,
                                          image_size=image_size)
    
    elapsed = time.time() - start_time
    
    print('Finished inference in {}'.format(humanfriendly.format_timespan(elapsed)))
