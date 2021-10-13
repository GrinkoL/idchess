import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential

class CreateModel():
    def __init__(self, base_model_name):
        if base_model_name == 'MobileNetV2':
            self.base_model = MobileNetV2(input_shape=(256,256,3), include_top=False)
        elif base_model_name == 'ResNet50':
            self.base_model = ResNet50(input_shape=(256,256,3),include_top=False)
        else:
            raise NameError('Not valid base model name. You may use ether "MobileNetV2" or ResNet50 names')
        self.input_layer = GrayscaleToRGBLayer(3,input_shape=(256,256,1))


    def assemble_model(self):
        '''
        '''
        for layer in self.base_model.layers[:-38]:
            layer.trainable = False
            
        model =  Sequential([self.input_layer, self.base_model, GlobalAveragePooling2D(),
                            Dense(128, activation='relu'),
                            Dense(128, activation='relu'),
                            Dense(8, activation=None)])
        return model


# Класс для перехода от черно-белого изображения с одним каналом к трёхканальному чёрно-белому изображению.
# Далее он будет использоваться как состовная часть нейросетевой модели.
class GrayscaleToRGBLayer(Layer): 
    def __init__(self, output_dim, **kwargs): 
        self.output_dim = output_dim 
        super(GrayscaleToRGBLayer, self).__init__(**kwargs) 
    def call(self, input_data):
        return tf.keras.backend.repeat_elements(input_data, self.output_dim, axis=3)