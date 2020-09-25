import tensorflow as tf
import numpy as np

def dynamic_rnn(rnn_type='lstm'):

    X=np.random.rand(3,6,4)
    X[1,4:]=0
    X_length=[6,4,6]

    rnn_hidden_size=5
    if(rnn_type=='lstm'):
        cell=tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size,state_is_tuple=True)
    else:
        cell=tf.contrib.rnn.GRUCell(num_units=rnn_hidden_size)

    num=cell.output_size

    outputs,last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float64,
        sequence_length=X_length,
        inputs=X
    )

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        o1,s1 = session.run([outputs,last_states])
        print(X)
        print(np.shape(o1))
        print(o1)
        print(np.shape(s1))
        print(s1)
        print(num)


if __name__ == '__main__':
    dynamic_rnn(rnn_type='lstm')