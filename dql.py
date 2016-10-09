import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[4])
target = tf.placeholder(tf.float32)

g=tf.reshape(x, [4,1])
x_t=tf.transpose(g)
W_fc1 = weight_variable([4, 10])
b_fc1 = bias_variable([10])
h_fc1 = tf.nn.sigmoid(tf.matmul(x_t, W_fc1) + b_fc1)

W_fc2 = weight_variable([10, 10])
b_fc2 = bias_variable([10])
h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([10, 2])
b_fc3 = bias_variable([2])

q_values = tf.matmul(h_fc2, W_fc3) + b_fc3

action  = tf.argmax(q_values, 1)
q_a_max = tf.reduce_max(q_values)

loss = tf.square(target- q_a_max)
with tf.name_scope('loss'):
    # loss = tf.clip_by_value(loss, 0, 1)
    tf.scalar_summary('TD Error', loss)

train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)

merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('./train', sess.graph)
sess.run(tf.initialize_all_variables())

def getAction(state):
    a = sess.run([action], feed_dict={x:state})
    a=int(a[0])
    return a

def learn(st, reward_p, state, i):
    for __ in range(20):
        q_t1_max = sess.run([q_a_max], feed_dict={x: state})
        t=reward_p + 0.9 * q_t1_max[0]
        gg, summary = sess.run([train_step, merged], feed_dict={target: t, x: st})
    train_writer.add_summary(summary, i)
