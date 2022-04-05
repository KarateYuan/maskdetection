import tengine as tg
import numpy as np


def load_tg_model(tg_model_path):
    graph = tg.Graph(None, 'tengine', tg_model_path)
    graph.preRun()
    return graph

def tg_inference(graph, img_arr):
    graph.run(block=1, input_data=img_arr)
    cls_tensor = graph.getOutputTensor(0, 0)
    bbox_tensor = graph.getOutputTensor(1, 0)
    # 这里会有内存泄漏吗
    bboxes = bbox_tensor.getbuffer(float)
    cls_scores = cls_tensor.getbuffer(float)

    cls_scores_t = bboxes[0:11944]
    bboxes_t = cls_scores[0:23888]
    
    cls_scores_nd = np.array(cls_scores_t).reshape((1, 5972, 2))
    bboxes_nd = np.array(bboxes_t).reshape((1, 5972, 4))
    return bboxes_nd, cls_scores_nd