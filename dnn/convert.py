import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

with tf.gfile.FastGFile('/home/mars/Mobilenet_v2/cnkrflags/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph_def = TransformGraph(graph_def, ['image_tensor'], ['detection_boxes', 'detection_classes', 'detection_scores', 'num_detections'], ['sort_by_execution_order'])
    with tf.gfile.FastGFile('/home/mars/Mobilenet_v2/cnkrflags/v2_cn_graph.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())
