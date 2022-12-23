import tensorflow as tf
import SequenceProcessing


def LSTM_V1(lstm_complexity=100, activation='swish'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(SequenceProcessing.SEQUENCE_LENGTH, SequenceProcessing.NUM_POINTS)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LSTM(lstm_complexity, dropout=0.4, return_sequences=False,
                                   input_shape=(SequenceProcessing.SEQUENCE_LENGTH, SequenceProcessing.NUM_POINTS)))
    
    model.add(tf.keras.layers.Dense(64, activation=activation))
    model.add(tf.keras.layers.Dense(128, activation=activation))
    model.add(tf.keras.layers.Dense(64, activation=activation))
    model.add(tf.keras.layers.Dense(32, activation=activation))
    model.add(tf.keras.layers.Dense(len(SequenceProcessing.classes), activation='softmax'))
    
    model.build()
    model.summary()
    return model


def LSTM_V2(lstm_complexity=25, activation='swish'):
    model = tf.keras.Sequential()

    input_net = tf.keras.layers.Input(shape=(SequenceProcessing.SEQUENCE_LENGTH, SequenceProcessing.NUM_POINTS))

    bn = tf.keras.layers.BatchNormalization()(input_net)
    pose, lh, rh = tf.split(bn, [SequenceProcessing.AMOUNT_BODY_LANDMARKS*3,
                                 SequenceProcessing.AMOUNT_HAND_LANDMARKS*3,
                                 SequenceProcessing.AMOUNT_HAND_LANDMARKS*3], axis=2)
    pose_lstm = tf.keras.layers.LSTM(lstm_complexity, dropout=0.4, return_sequences=False)(pose)
    lh_lstm = tf.keras.layers.LSTM(lstm_complexity, dropout=0.4, return_sequences=False)(lh)
    rh_lstm = tf.keras.layers.LSTM(lstm_complexity, dropout=0.4, return_sequences=False)(rh)

    merge = tf.keras.layers.Concatenate()([pose_lstm, lh_lstm, rh_lstm])

    dense0 = tf.keras.layers.Dense(64, activation=activation)(bn1)
    dense1 = tf.keras.layers.Dense(64, activation=activation)(dense0)
    drpout = tf.keras.layers.Dropout(0.3)(dense1)
    dense2 = tf.keras.layers.Dense(32, activation=activation)(drpout)
    out = tf.keras.layers.Dense(len(SequenceProcessing.classes), activation='softmax')(dense2)

    model = tf.keras.models.Model(inputs=input_net, outputs=out)
    model.summary()

    return model


def LSTM_V3(lstm_complexity=30, activation='swish'):
    model = tf.keras.Sequential()

    input_net = tf.keras.layers.Input(shape=(SequenceProcessing.SEQUENCE_LENGTH, SequenceProcessing.NUM_POINTS))

    bn = tf.keras.layers.BatchNormalization()(input_net)
    pose, lh, rh = tf.split(bn, [SequenceProcessing.AMOUNT_BODY_LANDMARKS*3,
                                 SequenceProcessing.AMOUNT_HAND_LANDMARKS*3,
                                 SequenceProcessing.AMOUNT_HAND_LANDMARKS*3], axis=2)
    pose_lstm = tf.keras.layers.LSTM(25, dropout=0.4, recurrent_dropout=0.3, return_sequences=False)(pose)
    lh_lstm = tf.keras.layers.LSTM(lstm_complexity, dropout=0.4,recurrent_dropout=0.3, return_sequences=False)(lh)
    rh_lstm = tf.keras.layers.LSTM(lstm_complexity, dropout=0.4, recurrent_dropout=0.3, return_sequences=False)(rh)

    merge = tf.keras.layers.Concatenate()([pose_lstm, lh_lstm, rh_lstm])

    bn0 = tf.keras.layers.BatchNormalization()(merge)

    dense1 = tf.keras.layers.Dense(63,activation=activation)(bn0)
    bn1 = tf.keras.layers.BatchNormalization()(dense1)

    dense2 = tf.keras.layers.Dense(63,activation=activation)(dense1)
    bn2 = tf.keras.layers.BatchNormalization()(dense2)

    dense3 = tf.keras.layers.Dense(63,activation=activation)(dense2)
    dropout1 = tf.keras.layers.Dropout(0.4)(dense3)
    bn3 = tf.keras.layers.BatchNormalization()(dropout1)

    add1 = tf.keras.layers.Add()([bn2, bn3])
    dense4 = tf.keras.layers.Dense(63,activation=activation)(add1)
    bn4 = tf.keras.layers.BatchNormalization()(dense4)

    add2 = tf.keras.layers.Add()([bn4, bn1])

    dense5 = tf.keras.layers.Dense(2*lstm_complexity+25,activation=activation)(add2)
    bn5 = tf.keras.layers.BatchNormalization()(dense5)
    add3 = tf.keras.layers.Add()([bn0, bn5])

    out = tf.keras.layers.Dense(len(SequenceProcessing.classes),activation='softmax')(add3) #, 

    model = tf.keras.models.Model(inputs=input_net, outputs=out)
    model.summary()

    return model


def LSTM_V4(lstm_complexity=25, activation='swish'):
    model = tf.keras.Sequential()

    input_net = tf.keras.layers.Input(shape=(SequenceProcessing.SEQUENCE_LENGTH, SequenceProcessing.NUM_POINTS))

    bn = tf.keras.layers.BatchNormalization()(input_net)
    lstm_1 = tf.keras.layers.LSTM(lstm_complexity, dropout=0.4, return_sequences=True)(bn)
    lstm_2 = tf.keras.layers.LSTM(lstm_complexity, dropout=0.4, return_sequences=True)(lstm_1)
    lstm_3 = tf.keras.layers.LSTM(lstm_complexity, dropout=0.4, return_sequences=False)(lstm_2)

    bn0 = tf.keras.layers.BatchNormalization()(lstm_3)

    dense1 = tf.keras.layers.Dense(63,activation=activation)(bn0)
    bn1 = tf.keras.layers.BatchNormalization()(dense1)

    dense2 = tf.keras.layers.Dense(63,activation=activation)(dense1)
    bn2 = tf.keras.layers.BatchNormalization()(dense2)

    dense3 = tf.keras.layers.Dense(63,activation=activation)(dense2)
    dropout1 = tf.keras.layers.Dropout(0.4)(dense3)
    bn3 = tf.keras.layers.BatchNormalization()(dropout1)

    add1 = tf.keras.layers.Add()([bn2, bn3])
    dense4 = tf.keras.layers.Dense(63,activation=activation)(add1)
    bn4 = tf.keras.layers.BatchNormalization()(dense4)

    add2 = tf.keras.layers.Add()([bn4, bn1])

    dense5 = tf.keras.layers.Dense(2*lstm_complexity+25,activation=activation)(add2)
    bn5 = tf.keras.layers.BatchNormalization()(dense5)
    add3 = tf.keras.layers.Add()([bn0, bn5])

    out = tf.keras.layers.Dense(len(SequenceProcessing.classes))(add3) #, 

    model = tf.keras.models.Model(inputs=input_net, outputs=out)
    model.summary()

    return model


def LSTM_V5(lstm_complexity=150, activation='swish'):
    model = tf.keras.Sequential()

    input_net = tf.keras.layers.Input(shape=(SequenceProcessing.SEQUENCE_LENGTH, SequenceProcessing.NUM_POINTS))

    bn = tf.keras.layers.BatchNormalization()(input_net)
    lstm_1 = tf.keras.layers.LSTM(lstm_complexity, dropout=0.4, return_sequences=False)(bn)

    bn0 = tf.keras.layers.BatchNormalization()(lstm_1)

    dense1 = tf.keras.layers.Dense(75,activation='tanh')(bn0)
    bn1 = tf.keras.layers.BatchNormalization()(dense1)

    dense2 = tf.keras.layers.Dense(75,activation=activation)(dense1)
    bn2 = tf.keras.layers.BatchNormalization()(dense2)

    dense3 = tf.keras.layers.Dense(75,activation=activation)(dense2)
    dropout1 = tf.keras.layers.Dropout(0.4)(dense3)
    bn3 = tf.keras.layers.BatchNormalization()(dropout1)

    add1 = tf.keras.layers.Add()([bn2, bn3])
    dense4 = tf.keras.layers.Dense(75,activation=activation)(add1)
    bn4 = tf.keras.layers.BatchNormalization()(dense4)

    add2 = tf.keras.layers.Add()([bn4, bn1])

    dense5 = tf.keras.layers.Dense(lstm_complexity,activation=activation)(add2)
    bn5 = tf.keras.layers.BatchNormalization()(dense5)
    add3 = tf.keras.layers.Add()([bn0, bn5])

    out = tf.keras.layers.Dense(len(SequenceProcessing.classes),activation='softmax')(add3) #, 

    model = tf.keras.models.Model(inputs=input_net, outputs=out)
    model.summary()

    return model


