# This transformer is based on paper "Attention is all you need. In pytorch it's available as an nn package but in keras need to implement it.

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

d_model = 256
max_length = 1012 #maximum source seq lenth
num_layers_encoder = 6
num_decoder_layer = 6

#creating positional encoding
def positional_encoding(pos, d_model):
    """ Compute positional encoding for a particular position
    Args:
        pos: position of a token in the sequence
        d_model: depth size of the model
    
    Returns:
        The positional encoding for the given token
    """
    PE = np.zeros((1, d_model))
    for i in range(d_model):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 10000 ** (i / d_model))
        else:
            PE[:, i] = np.cos(pos / 10000 ** ((i - 1) / d_model))
    return PE

pes = []
for i in range(max_length):
    pes.append(positional_encoding(i,d_model))

pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)
# print(pes.shape)

"""Create the Multihead Attention layer"""
class MultiHeadAttention(tf.keras.Model):
    """ Class for Multi-Head Attention layer
    Attributes:
        key_size: d_key in the paper
        h: number of attention heads
        wq: the Linear layer for Q
        wk: the Linear layer for K
        wv: the Linear layer for V
        wo: the Linear layer for the output
    """
    def __init__(self, d_model, h):
        super(MultiHeadAttention, self).__init__()
        self.key_size = d_model // h
        self.h = h
        self.wq = tf.keras.layers.Dense(d_model) 
        self.wk = tf.keras.layers.Dense(d_model) 
        self.wv = tf.keras.layers.Dense(d_model) 
        self.wo = tf.keras.layers.Dense(d_model)
        self.reshape1 = tf.keras.layers.Reshape((-1,self.h,self.key_size))
        self.reshape2 = tf.keras.layers.Reshape((-1,self.h*self.key_size))

    def call(self, query, value, mask=None):
        """ The forward pass for Multi-Head Attention layer
        Args:
            query: the Q matrix
            value: the V matrix, acts as V and K
            mask: mask to filter out unwanted tokens
                  - zero mask: mask for padded tokens
                  - right-side mask: mask to prevent attention towards tokens on the right-hand side
        
        Returns:
            The concatenated context vector
            The alignment (attention) vectors of all heads
        """
        # query has shape (batch, query_len, d_model)
        # value has shape (batch, value_len, d_model)
        query = self.wq(query)
        key = self.wk(value)
        value = self.wv(value)
        
        # Split matrices for multi-heads attention
        # batch_size = query.get_shape().as_list()[0]
        
        # Originally, query has shape (batch, query_len, d_model)
        # We need to reshape to (batch, query_len, h, key_size)
        query = self.reshape1(query)
        # In order to compute matmul, the dimensions must be transposed to (batch, h, query_len, key_size)
        query = tf.transpose(query, [0, 2, 1, 3])
        
        # Do the same for key and value
        key = self.reshape1(key)
        key = tf.transpose(key, [0, 2, 1, 3])
        value = self.reshape1(value)
        value = tf.transpose(value, [0, 2, 1, 3])
        
        # Compute the dot score
        # and divide the score by square root of key_size (as stated in paper)
        # (must convert key_size to float32 otherwise an error would occur)
        score = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))
        # score will have shape of (batch, h, query_len, value_len)
        
        # Mask out the score if a mask is provided
        # There are two types of mask:
        # - Padding mask (batch, 1, 1, value_len): to prevent attention being drawn to padded token (i.e. 0)
        # - Look-left mask (batch, 1, query_len, value_len): to prevent decoder to draw attention to tokens to the right
        if mask is not None:
            score *= mask

            # We want the masked out values to be zeros when applying softmax
            # One way to accomplish that is assign them to a very large negative value
            score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)
        
        # Alignment vector: (batch, h, query_len, value_len)
        alignment = tf.nn.softmax(score, axis=-1)
        
        # Context vector: (batch, h, query_len, key_size)
        context = tf.matmul(alignment, value)
        
        # Finally, do the opposite to have a tensor of shape (batch, query_len, d_model)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = self.reshape2(context)
        
        # Apply one last full connected layer (WO)
        heads = self.wo(context)
        
        return heads, alignment

class Encoder(tf.keras.Model):
    """ Class for the Encoder
    Args:
        d_model: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        embedding: Embedding layer
        embedding_dropout: Dropout layer for Embedding
        attention: array of Multi-Head Attention layers
        attention_dropout: array of Dropout layers for Multi-Head Attention
        attention_norm: array of LayerNorm layers for Multi-Head Attention
        dense_1: array of first Dense layers for FFN
        dense_2: array of second Dense layers for FFN
        ffn_dropout: array of Dropout layers for FFN
        ffn_norm: array of LayerNorm layers for FFN
    """
    def __init__(self,d_model, num_layers, h):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.h = h
        # self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.sequence_dropout = tf.keras.layers.Dropout(0.1)
        self.attention = [MultiHeadAttention(d_model, h) for _ in range(num_layers)]
        self.attention_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]

        self.attention_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            d_model * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            d_model) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

    def call(self, sequence, training=True, encoder_mask=None):
        """ Forward pass for the Encoder
        Args:
            sequence: source input sequences
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention
        
        Returns:
            The output of the Encoder (batch_size, length, d_model)
            The alignment (attention) vectors for all layers
        """
        # embed_out = self.embedding(sequence)

        # embed_out *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embed_out = sequence + pes[:sequence.shape[1], :]
        embed_out = self.sequence_dropout(embed_out)

        sub_in = embed_out
        alignments = []

        for i in range(self.num_layers):
            sub_out, alignment = self.attention[i](sub_in, sub_in, encoder_mask)
            sub_out = self.attention_dropout[i](sub_out, training=training)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)
            
            alignments.append(alignment)
            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out

        return ffn_out, alignments


""" creating decoder """
class Decoder(tf.keras.Model):
    """ Class for the Decoder
    Args:
        d_model: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        embedding: Embedding layer
        embedding_dropout: Dropout layer for Embedding
        attention_bot: array of bottom Multi-Head Attention layers (self attention)
        attention_bot_dropout: array of Dropout layers for bottom Multi-Head Attention
        attention_bot_norm: array of LayerNorm layers for bottom Multi-Head Attention
        attention_mid: array of middle Multi-Head Attention layers
        attention_mid_dropout: array of Dropout layers for middle Multi-Head Attention
        attention_mid_norm: array of LayerNorm layers for middle Multi-Head Attention
        dense_1: array of first Dense layers for FFN
        dense_2: array of second Dense layers for FFN
        ffn_dropout: array of Dropout layers for FFN
        ffn_norm: array of LayerNorm layers for FFN
        dense: Dense layer to compute final output
    """
    def __init__(self, d_model, num_layers, h):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.h = h
        # self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.attention_bot = [MultiHeadAttention(d_model, h) for _ in range(num_layers)]
        self.attention_bot_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(d_model, h) for _ in range(num_layers)]
        self.attention_mid_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            d_model * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            d_model) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(0.1) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        # self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, encoder_output, training=True, encoder_mask=None):
        """ Forward pass for the Decoder
        Args:
            sequence: source input sequences
            encoder_output: output of the Encoder (for computing middle attention)
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention
        
        Returns:
            The output of the Encoder (batch_size, length, d_model)
            The bottom alignment (attention) vectors for all layers
            The middle alignment (attention) vectors for all layers
        """
        # EMBEDDING AND POSITIONAL EMBEDDING
        # embed_out = self.embedding(sequence)

        # embed_out *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embed_out = sequence + pes[:sequence.shape[1], :]
        embed_out = self.embedding_dropout(embed_out)

        bot_sub_in = embed_out
        bot_alignments = []
        mid_alignments = []

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            seq_len = bot_sub_in.shape[1]

            if training:
                mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            else:
                mask = None
            bot_sub_out, bot_alignment = self.attention_bot[i](bot_sub_in, bot_sub_in, mask)
            bot_sub_out = self.attention_bot_dropout[i](bot_sub_out, training=training)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)
            
            bot_alignments.append(bot_alignment)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out, mid_alignment = self.attention_mid[i](
                mid_sub_in, encoder_output, encoder_mask)
            mid_sub_out = self.attention_mid_dropout[i](mid_sub_out, training=training)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)
            
            mid_alignments.append(mid_alignment)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        # logits = self.dense(ffn_out)
        logits = ffn_out

        return logits, bot_alignments, mid_alignments

class DETR(tf.keras.Model):
    """ Class for the DETR
    Args:
        d_model: d_model in the paper (depth size of the model) (default = 256)
        num_decoder_layer: number of layer in decoder (default = 6)
        num_encoder_layer: number of layer in encoder (default = 6)
        nhead: number of heads in multiattention layer (default = 8)
        num_classes: number of classes in output (default = 1000)
        num_query: number of queries in output (default = 100)
    """
    def __init__(self,d_model = 256,nhead = 8,num_decoder_layer = 6,num_encoder_layer = 6,num_classes = 1000,num_query = 100):
        super(DETR, self).__init__()
        self.d_model = d_model
        self.backbone = ResNet50(include_top = False,input_shape=(224,224,3),weights=None)
        self.conv2d = tf.keras.layers.Conv2D(d_model,1)
        self.encoder = Encoder(d_model,num_encoder_layer,nhead)
        self.decoder = Decoder(d_model,num_decoder_layer,nhead)
        self.linear_class = tf.keras.layers.Dense(num_classes+1,activation='softmax')
        self.linear_bbox = tf.keras.layers.Dense(4,activation='sigmoid')
        self.query_pos = tf.constant(tf.random.uniform(shape=(max_length,num_query,d_model)))

    def call(self,inputs,training=True):
        """ Forward pass for the Decoder
        Args:
            inputs: source input
            training: whether training or not (for Dropout)
        Returns:
            The output of class prediction
            The output of bboxes
        """
        x = self.backbone(inputs)
        x = self.conv2d(x)
        print(x.shape)
        size = x.get_shape().as_list()
        x = tf.keras.layers.Reshape((size[1]*size[2],self.d_model))(x)
        print(x.shape)
        encoder_output, _ = self.encoder(x,training=training)
        print(encoder_output.shape)
        # stacking query_pos batch_size times
        logits, _, _ = self.decoder(self.query_pos[:tf.shape(encoder_output)[0],:,:],encoder_output,training=training)
        # logits = encoder_output
        pred_logits = self.linear_class(logits)
        pred_boxes = self.linear_bbox(logits)
        return pred_logits,pred_boxes

detr = DETR()
inputs =  tf.keras.Input(shape=(224,224,3))
outputs, _ = detr(inputs)
model = tf.keras.Model(inputs=inputs,outputs=outputs)
model.summary()