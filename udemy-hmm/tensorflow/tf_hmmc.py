import wave
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
MVN = tf.contrib.distributions.MultivariateNormalDiag
from generate_c import get_signals, big_init

class HMM:
    def __init__(self, M, K, D):
        self.M = M
        self.K = K
        self.D = D

    def set_session(self, session):
        self.session  = session

    def init_random(self, X):
        pi0 = np.ones(self.M).astype(np.float32)
        A0 = np.random.randn(self.M, self.M).astype(np.float32)
        R0 = np.ones((self.M, self.K)).astype(np.float32)

        mu0 = np.zeros((self.M, self.K, self.D))
        for j in range(self.M):
            for k in range(self.K):
                n = np.random.randint(X.shape[0])
                t = np.random.randint(X.shape[1])
                mu0[j,k] = X[n,t]

        mu0 = mu0.astype(np.float32)

        sigma0 = np.random.randn((self.M, self.K, self.D)).astype(np.float32)

    def build(self, preSoftmaxPi, preSoftmaxA, preSoftmaxR, mu, logSigma):
        self.preSoftmaxPi = tf.Variable(preSoftmaxPi)
        self.preSoftmaxA = tf.Variable(preSoftmaxA)
        self.preSoftmaxR = tf.Variable(preSoftmaxR)
        self.mu = tf.Variable(mu)
        self.logSigma = tf.Variable(logSigma)

        pi = tf.nn.softmax(self.preSoftmaxPi)
        A = tf.nn.softmax(self.preSoftmaxA)
        R = tf.nn.softmax(self.preSoftmaxR)
        sigma = tf.exp(self.logSigma)

        # X will be T x D
        self.tfx = tf.placeholder(tf.float32, shape=(None, self.D), name='X')

        self.mvns = []
        for j in range(self.M):
            self.mvns.append([])
            for k in range(self.K):
                self.mvns[j].append(
                    MVN(self.mu[j,k], sigma[j,k])
                )

        B = []
        for j in range(self.M):
            components = []
            for k in range(self.K):
                components.append(
                    self.mvns[j][k].prob(self.tfx)
                )

            components = tf.stack(components)
            R_j = tf.reshape(R[j], [1, self.K])
            p_x_t_j = tf.matmul(R_j, components)

            components = tf.reshape(p_x_t_j, [-1])

            B.append(components)

        B = tf.stack(B)

        B = tf.transpose(B, [1,0])

        def recurrence(old_a_old_s, B_t):
            old_a = tf.reshape(old_a_old_s[0], (1, self.M))
            a = tf.matmul(old_a, A) * B_t
            a = tf.reshape(a,(self.M,))
            s = tf.reduce_sum(a)
            return (a / s), s

        alpha, scale = tf.scan(
            fn = recurrence,
            elems = B[1:],
            initializer=(pi*B[0], np.float32(1.0))
        )
        self.cost_op = -tf.reduce_sum(tf.log(scale))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.cost_op)

    def set(self, preSoftmaxPi, preSoftmaxA, preSoftmaxR, mu, logSigma):
        op1 = self.preSoftmaxPi.assign(preSoftmaxPi)
        op2 = self.preSoftmaxA.assign(preSoftmaxA)
        op3 = self.preSoftmaxR.assign(preSoftmaxR)
        op4 = self.mu.assign(mu)
        op5 = self.logSigma.assign(logSigma)

        self.session.run([op1, op2, op3, op4, op5])

    def fit(self, X, max_iter=10):
        N = len(X)
        print("num of train samples:",N)

        costs = []
        for it in xrange(max_iter):
            if it% 1 == 0:
                print "it: ", it

            for n in xrange(N):
                c = self.get_cost_multi(X).sum()
                costs.append(c)
                self.session.run(self.train_op, feed_dict={self.tfx: X[n]})

    def get_cost(self, x):
        return self.session.run(self.cost_op, feed_dict={self.tfx: x})

    def get_cost_multi(self, X):
        return np.array(]self.get_cost(x) for x in X])

def real_signal():
    spf = wave.open('', 'r')

    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    T = len(signal)
    signal = (signal - signal.mean())/ signal.std()

    signals = signal.reshape(1, T, 1)

    hmm = HMM(3,3,1)
    hmm.init_random(signals)

    init = tf.global_variables.initializer()
    session = tf.InteractiveSession()

    hmm.fit(signals, max_iter = 30)
