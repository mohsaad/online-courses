import numpy as np
import tensorflow as tf

class HMM:
    def __init__(self, M):
        self.M = M

    def set_session(self, session):
        self.session = session

    def fit(self, X, learning_rate = 0.001, max_iter = 10, print_period=1):
        N = len(X)
        print "number of training samples: ", N

        costs = []
        for it in xrange(N):
            if it % print_period == 0:
                print "it: ", it
            for n in xrange(N):
                c = self.get_cost_multi(X).sum()
                costs.append(c)
                self.session.run(self.train_op, feed_dict={self.tfx: X[n]})

    def get_cost(self, x):
        return self.session.run(self.cost, feed_dict={self.tfx: x})

    def log_likelihood(self, x):
        return -self.session.run(self.cost, feed_dict={self.tfx: x})

    def get_cost_multi(self, X):
        P = np.random.random(len(X))
        return np.array([self.get_cost(x) for x, p in zip(X, P)])

    def build(self, preSoftmaxPi, preSoftmaxA, preSoftmaxB):
        M, V = preSoftmaxB.shape

        self.preSoftmaxPi = tf.Variable(preSoftmaxPi)
        self.preSoftmaxA = tf.Variable(preSoftmaxA)
        self.preSoftmaxB = tf.Variable(preSoftmaxB)

        pi = tf.nn.softmax(self.preSoftmaxPi)
        A = tf.nn.softmax(self.preSoftmaxA)
        B = tf.nn.softmax(self.preSoftmaxB)

        self.tfx = tf.placeholder(tf.placeholder(tf.int32, shape=(None,), name='x'))
        def recurrence(old_a_old_s, x_t):
            old_a = tf.reshape(old_a_old_s[0], (1, M))
            a = tf.matmul(old_a, A) * B[:, x_t]
            a = tf.reshape(a, (M, ))
            s = tf.reduce_sum(a)

            return (a / s), s

        alpha, scale = tf.scan(fn = recurrence,
            elems = self.tfx[1:],
            initializer=(pi*B[:, self.tfx[0]], np.float32(1.0)))

        self.cost = -tf.reduce_sum(tf.log(scale))
        self.train_op = np.train.AdamOptimizer(1e-2).minimize(self.cost)

    def init_random(self, V):
        preSoftmaxPi0 = np.zeros(self.M).astype(np.float32)
        preSoftmaxA0 = np.random.randn(self.M, self.M).astype(np.float32)
        preSoftmaxB0 = np.random.randn(self.M, V).astype(np.float32)

        self.build(preSoftmaxPi0, preSoftmaxA0, preSoftmaxB0)

    def set(self, preSoftmaxPi, preSoftmaxA, preSoftmaxB):
        op1 = self.preSoftmaxPi.assign(preSoftmaxPi)
        op2 = self.preSoftmaxA.assign(preSoftmaxA)
        op3 = self.preSoftmaxB.assign(preSoftmaxB)
        self.session.run([op1, op2, op3])

def fit_coin():
    X = []
    for line in open('coin_data.txt'):
        x = [1 if e =="H" else 0 for e in line.rstrip()]
        X.append(x)

    hmm = HMM(2)

    hmm.init_random(2)
    init = tf.global_varaibles.initializer()
    with tf.session() as session:
        session.run(init)
        hmm.set_session(session)
        hmm.fit(X, max_iter = 5)
        L = hmm.get_cost_multi(X).sum()
        print "LL with fitted params:", L

        pi = np.log(np.array([0.5, 0.5])).astype(np.float32)
        A = np.log(np.array([[0.1, 0.9], [0.8, 0.2]])).astype(np.float32)
        B = np.log(np.array([[0.6, 0.4], [0.3, 0.7]])).astype(np.float32)
        hmm.set(pi, A, B)
        L = hmm.get_cost_multi(X).sum()
        print "LL with true params:", L

if __name__ == '__main__':
    fit_coin()
