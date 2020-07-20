from keras.layers import ZeroPadding2D
from keras.models import Model
from keras_resnet import models as resnet_models
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf
from ..utils import centernet_loss, decode
from keras.layers import (Input,
                          Conv2DTranspose,
                          BatchNormalization,
                          ReLU,
                          Conv2D,
                          Lambda,
                          MaxPooling2D,
                          Dropout)

class CenterNet():
    def __init__(self, 
                 num_classes, 
                 backbone_resnet='resnet50', 
                 input_size=512, 
                 max_objects=100, 
                 score_threshold=0.1,
                 nms=True,
                 flip_test=False,
                 mode="train"
                ):
        
        self.nms = nms
        self.flip_test = flip_test
        self.score_threshold = score_threshold
        self.max_objects = max_objects
        self.output_size = input_size // 4
        self.image_input = Input(shape=(None, None, 3))
        self.heatmap_input = Input(shape=(self.output_size, self.output_size, num_classes))
        self.width_height_input = Input(shape=(max_objects, 2))
        self.offset_input = Input(shape=(max_objects, 2))
        self.offset_mask_input = Input(shape=(max_objects,))
        self.index_input = Input(shape=(max_objects,))
        self.num_classes = num_classes
        self.backbone = resnet_models.ResNet50(self.image_input, include_top=False) if backbone_resnet=='resnet50' else resnet_models.ResNet101(image_input, include_top=False)
        
        #Pass the input through resnet backbone and take the output after conv5
        
        
        
    def build_model(self):
        self.output_conv_5 = self.backbone.outputs[-1]
        self.x = Dropout(rate=0.5)(self.output_conv_5)
        num_filters = 256
        for i in range(3):
            num_filters = num_filters // pow(2,i)
            self.x = Conv2DTranspose(
                num_filters,
                (4, 4),
                strides = 2,
                use_bias = False,
                padding = 'same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(5e-4)
            )(self.x)
            
            self.x = BatchNormalization()(self.x)
            self.x = ReLU()(self.x)
            
        return self.x
        
    
    
    
    def return_model(self, outputs,detections):
        
        inputs = [self.image_input, self.heatmap_input, self.width_height_input, self.offset_input, self.offset_mask_input, self.index_input]
        model = Model(inputs = inputs, outputs = outputs)
        
        prediction_model = Model(inputs=self.image_input, outputs=detections)
        
        return model, prediction_model
            
    def heatmap_header(self, x):#heatmap
        y = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Conv2D(self.num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y)
        
        return y
    def width_height_header(self, x):#embedding
            
        y = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y)
        
        return y
    
    def offset_header(self, x):#offsets
        y = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4))(x)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), activation='sigmoid')(y)
        
        return y
    
    @staticmethod        
    def compute_loss(self):
        
        self.x = self.build_model()
        y1 = self.heatmap_header(self.x)
        y2 = self.width_height_header(self.x)
        y3 = self.offset_header(self.x)
        
        outputs =  Lambda(centernet_loss, name="centernet_loss")([y1, y2, y3, self.heatmap_input, self.width_height_input, self.offset_input, self.offset_mask_input, self.index_input])
        detections = Lambda(lambda x: decode(*x,
                                         max_objects=self.max_objects,
                                         score_threshold=self.score_threshold,
                                         nms=self.nms,
                                         flip_test=self.flip_test,
                                         num_classes=self.num_classes))([y1, y2, y3])
        
        return outputs, detections
    
    def forward(self):
        outputs, detections = self.compute_loss(self)
        model, prediction_model = self.return_model(outputs, detections)
        return model, prediction_model
        
        
## need to check if these functions make the graph fully trainable