import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.int32, shape=(None,), name='x')

def square(last, current):
    return current * current

def fib(last, current):
    return (last[1], last[0] + last[1])



square_op = tf.scan(
fn = fib, elems = x, initializer = (0,1)
)

with tf.Session() as session:
    o_val = session.run(square_op, feed_dict={x: [1,2,3,4,5]})
    print("output", o_val)
