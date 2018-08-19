from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from moviepy.editor import VideoFileClip


# Frozen inference graph files. NOTE: change the path to where you saved the models.
SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
SSD_v2_GRAPH_FILE = 'ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
SSDLITE_GRAPH_FILE = 'ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
RFCN_GRAPH_FILE = 'rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'
FASTER_RCNN_GRAPH_FILE = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'


# detection_graph = load_graph(SSD_GRAPH_FILE)
# detection_graph = load_graph(RFCN_GRAPH_FILE)
# detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)
# detection_graph = load_graph(SSDLITE_GRAPH_FILE)
detection_graph = load_graph(SSD_v2_GRAPH_FILE)

# The input placeholder for the image.
# `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Each box represents a part of the image where a particular object was detected.
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

# The classification of the object (integer id).
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')


def verify_on_sample_image():
    image = Image.open('./assets/sample1.jpg')
    image_np = np.expand_dims(np.asanyarray(image, dtype=np.uint8), 0)

    with tf.Session(graph=detection_graph) as sess:
        # Actual detection.
        (boxes, scores, classes) = sess.run([detection_boxes,
                                             detection_scores,
                                             detection_classes],
                                            feed_dict={image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.8

        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = image.size
        box_coords = to_image_coords(boxes, height, width)

        # Each class with be represented by a differently colored box
        draw_boxes(image, box_coords, classes)

        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.show()


tf_session = 0


def pipeline(img):
    global tf_session

    image_np = np.expand_dims(img, 0)
    (boxes, scores, classes) = tf_session.run([detection_boxes,
                                               detection_scores,
                                               detection_classes],
                                              feed_dict={image_tensor: image_np})

    # Remove unnecessary dimensions
    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)

    confidence_cutoff = 0.8

    # Filter boxes with a confidence score less than `confidence_cutoff`
    boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

    draw_img = Image.fromarray(img)
    # The current box coordinates are normalized to a range between 0 and 1.
    # This converts the coordinates actual location on the image.
    width, height = draw_img.size
    box_coords = to_image_coords(boxes, height, width)

    # Each class with be represented by a differently colored box
    draw_boxes(draw_img, box_coords, classes)
    return np.array(draw_img)


def process_video():
    global tf_session
    clip = VideoFileClip('driving.mp4')

    with tf.Session(graph=detection_graph) as sess:
        tf_session = sess
        new_clip = clip.fl_image(pipeline)

        # write to file
        new_clip.write_videofile('result.mp4')


if __name__ == "__main__":
    # verify_on_sample_image()
    process_video()