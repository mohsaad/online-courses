import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = tf.placeholder(tf.int32, shape=(None,), name='x')

signal = np.sin(np.linspace(0, 2*np.pi, 201))
corrupt = 2*np.random.randn(201) + signal

decay = tf.placeholder(tf.float64, shape=(), name='decay')
sequence = tf.placeholder(tf.float64, shape=(None,), name='sequence')

decay_rate = 0.97

def square(last, current):
    return current * current

def fib(last, current):
    return (last[1], last[0] + last[1])

def lpf(last, current):
    return decay * last + (1 - decay_rate) * current

square_op = tf.scan(
fn = lpf, elems = sequence, initializer = sequence[0]
)

with tf.Session() as session:
    Y = session.run(square_op, feed_dict={sequence: corrupt, decay: decay_rate})

print(np.power(Y - signal, 2).sum(), np.power(corrupt - signal, 2).sum())
# plt.plot(Y)
# plt.plot(signal)
# plt.plot(corrupt)
# plt.title("All our signals")
# plt.savefig('test.png')
