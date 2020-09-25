import  tensorflow as tf
import numpy as np
import random
from collections import  deque

GAMMA = 0.8
OBSERVE = 300
EXPLORE = 180000
FINAL_EPSILON = 0.0
INITIAL_EPSILON = 0.8
REPLAY_MEMORY = 400
BATCH_SIZE = 128
STEP_SIZE = 1

class BrainDRQN:

    def __init__(self,actionnum,sensornum):

        self.actionnum = actionnum
        self.sensornum = sensornum
        self.replaymemory = deque()
        self.epsilon = INITIAL_EPSILON
        self.stepsize = STEP_SIZE
        self.timestep = 0
        self.hidden1 = 64
        self.hidden2 = 128
        self.hidden3 = 128
        self.createQNetwork()
        self.stateflag = 0
        self.tempmat = []

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def createQNetwork(self):

        self.stateInput = tf.placeholder("float",[None,self.stepsize,self.sensornum])

        lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden1,name='q_lstm')
        lstm_out, state = tf.nn.dynamic_rnn(lstm,self.stateInput,dtype=tf.float32)
        reduced_out = lstm_out[:,-1,:]
        reduced_out = tf.reshape(reduced_out, shape=[-1,self.hidden1])

        W_fc1 = self.weight_variable([self.hidden1, self.hidden1])
        b_fc1 = self.bias_variable([self.hidden1])

        W_fc2 = self.weight_variable([self.hidden1, self.hidden2])
        b_fc2 = self.bias_variable([self.hidden2])

        W_fc3 = self.weight_variable([self.hidden2, self.hidden3])
        b_fc3 = self.bias_variable([self.hidden3])

        W_fc4 = self.weight_variable([self.hidden3, self.actionnum])
        b_fc4 = self.bias_variable([self.actionnum])

        h_fc1 = tf.nn.relu(tf.matmul(reduced_out, W_fc1) + b_fc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_fc3 = tf.nn.tanh(tf.matmul(h_fc2, W_fc3) + b_fc3)

        self.QValue = tf.matmul(h_fc3, W_fc4) + b_fc4

        self.actionInput = tf.placeholder("float", [None, self.actionnum])
        self.QValue_T = tf.placeholder("float", [None])

        Q_action = tf.multiply(self.QValue, self.actionInput)
        Q_action = tf.reduce_sum(Q_action, reduction_indices=1)

        self.cost = tf.reduce_mean(tf.square(self.QValue_T - Q_action))
        self.trainStep = tf.train.AdamOptimizer(learning_rate=10 ** -5).minimize(self.cost)

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def getAction(self,actionnum,stateInput,time):
        action = np.zeros(actionnum)
        if time <=2100:#增加初始随机步骤，先填充满replaymemory
            action_index = random.randrange(actionnum)
            action[action_index] = 1
            print("use random strategy:")
            print(action_index)
        else:
            if random.random() <= self.epsilon:
                action_index = random.randrange(actionnum)
                action[action_index] = 1
                print("use random strategy:")
                print(action_index)
            else:
                Qvalue = self.QValue.eval(feed_dict={self.stateInput:stateInput})
                print("use max Q-value:")
                action_index = np.argmax(Qvalue)
                print([action_index])
                action[action_index] = 1
            if self.epsilon > FINAL_EPSILON and self.timestep > OBSERVE:
                self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action


    def trainQnetwork(self):
        minibatch = random.sample(self.replaymemory,BATCH_SIZE)

        minibatch = np.array(minibatch)

        #print("minibatch size:")

        #print(np.shape(minibatch))

        #print("minibatch is:")

        #print(minibatch)

        statebatch = []
        for i in range(BATCH_SIZE):
            for j in range(self.stepsize):
                statebatch.append(minibatch[i][j][0])
        statebatch = np.reshape(statebatch,[BATCH_SIZE,self.stepsize,self.sensornum])

        rewardbatch = []
        for i in range(BATCH_SIZE):
            rewardbatch.append(minibatch[i][self.stepsize-1][3])

        actionbatch = []
        for i in range(BATCH_SIZE):
            actionbatch.append(minibatch[i][self.stepsize-1][2])

        newstatebatch = []
        for i in range(BATCH_SIZE):
            for j in range(self.stepsize):
                newstatebatch.append(minibatch[i][j][1])
        newstatebatch = np.reshape(newstatebatch, [BATCH_SIZE, self.stepsize, self.sensornum])

        Qvalue_T_batch = []
        Qvalue_batch = self.QValue.eval(feed_dict={self.stateInput:newstatebatch})


        print("train Q network......")
        print("-------------------------")

        for i in range(BATCH_SIZE):
            Qvalue_T_batch.append(rewardbatch[i] + GAMMA * np.max(Qvalue_batch))

        _,self.loss = self.session.run([self.trainStep,self.cost],feed_dict={
                        self.actionInput : actionbatch,
                        self.stateInput : statebatch,
                        self.QValue_T : Qvalue_T_batch})

        print("loss is %d" %self.loss)

        return self.loss

    def getLoss(self):
        loss = 0
        if len(self.replaymemory) > REPLAY_MEMORY:
            self.replaymemory.popleft()
        if self.timestep > OBSERVE:
            loss = self.trainQnetwork()
        self.timestep += 1

        return loss

    def sendmemory(self,currentstate,nextstate,action,reward):
        if self.stateflag < self.stepsize:
            #print("这里经过了，stateflag：")
            #print(self.stateflag)
            self.tempmat.append((currentstate,nextstate,action,reward))
            self.stateflag = self.stateflag + 1
            #print(self.stateflag)

        else:
            self.replaymemory.append(self.tempmat)
            #print("show tempmat:")
            #print(self.tempmat)
            self.tempmat = []
            self.stateflag = 0




































