from shapely.geometry import box, Polygon
import tensorflow as tf
import numpy as np

def loss_box(y_true, y_pred):
    return tf.losses.MSE(y_true, y_pred)

def meanIoU(y_true, y_pred):
    '''
    Делегирует вычисления meanIoU на функцию calculate_meanIoU для того,
    чтобы работать с numpy массивами, а не с EagerTensor
    '''
    mIoU = tf.py_function(calculate_meanIoU, [y_true, y_pred], tf.float32)
    return mIoU

def calculate_meanIoU(y_true_matrix, y_pred_matrix):
    '''
    Вычисляет метрику mIoU в процентах

    Параметры
    --------------------
    y_true: numpy.ndarray
        Двумерный массив, нулевое измерение которого равно количеству элементов
        в батче, а первое - 8ти
    y_pred: numpy.ndarray
        Аналогично y_true

    Возвращаемое/мые значения
    --------------------
    :float
        Значение метрики mIoU в процентах
    '''
    iou_list = []
    for y_true, y_pred in zip(y_true_matrix, y_pred_matrix):
        quad_true_xy = [[y_true[0], y_true[1]], [y_true[2], y_true[3]], [y_true[4], y_true[5]], [y_true[6], y_true[7]]]
        quad_pred_xy = [[y_pred[0], y_pred[1]], [y_pred[2], y_pred[3]], [y_pred[4], y_pred[5]], [y_pred[6], y_pred[7]]]
        quad_true_shape = Polygon(quad_true_xy)
        quad_pred_shape = Polygon(quad_pred_xy)
        if quad_pred_shape.is_valid: # Если нет самопересечений
            polygon_intersection = quad_true_shape.intersection(quad_pred_shape).area
            polygon_union = quad_true_shape.union(quad_pred_shape).area
            iou = polygon_intersection / polygon_union
        else: # при обнаружении самопересечений зануляем метрику IoU
            iou = 0
        iou_list.append(iou)
    meanIoU = np.mean(iou_list)
    return meanIoU * 100