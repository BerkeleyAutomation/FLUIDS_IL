import numpy as np
import argparse
import environments
import pickle
import tensorflow as tf

#def normc_initializer(std=1.0):
#    def _initializer(shape, dtype=None, partition_info=None):
#        out = np.random.randn(*shape).astype(np.float32)
#        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
#        return tf.constant(out)
#
#    return _initializer
#
#def create_model(input_dim, num_outputs):
#    obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, *input_dim])
#    last_layer = tf.layers.conv2d(
#            obs_ph,
#            32,
#            (4, 4),
#            activation=tf.nn.relu)
#
#    last_layer = tf.contrib.layers.flatten(last_layer)
#    last_layer = tf.layers.dense(
#            last_layer,
#            256,
#            kernel_initializer=normc_initializer(0.01),
#            activation = tf.nn.relu)
#    output = tf.layers.dense(
#            last_layer,
#            num_outputs,
#            kernel_initializer=normc_initializer(0.01),
#            activation = None)
#    return obs_ph, output
#
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("env_name", type=str, help="name of env",
#                        choices=["FluidsEnv", "FluidsPixelEnv",
#                                 "FluidsEdgeEnv", "FluidsVelEnv"])
#    parser.add_argument("--train_steps", type=int, help="number of steps to train",
#                        default = 1000)
#    parser.add_argument("--expert_data", type=str, help="rollouts from supervisor",
#                        default = "rollouts.pkl")
#    parser.add_argument('--test_eps', type=int, default=5)
#    parser.add_argument("--output_file", type=str, help="name of output file",
#                        default="rewards.pkl")
#    args = parser.parse_args()
#    env_name = args.env_name
#    env = getattr(environments, env_name)()
#
#    expert_file = open(args.expert_data, 'rb')
#    expert_data = pickle.load(expert_file)
#
#    with tf.Session() as sess:
#        obs_size = env.observation_space.shape
#        action_size = 1 if env.action_space.shape is () else env.action_space.shape[0]
#        print (obs_size, action_size)
#        print (expert_data['observations'].shape)
#        print (expert_data['actions'].shape)
#
#        input_ph, output_pred = create_model(obs_size, action_size)
#        output_ph = tf.placeholder(dtype=tf.float32, shape=[None, action_size])
#        mse = tf.reduce_mean(tf.square(output_pred - output_ph))
#        opt = tf.train.AdamOptimizer().minimize(mse)
#        sess.run(tf.global_variables_initializer())
#        reward_means = []
#        reward_stds = []
#        for train_step in range(args.train_steps):
#            sess.run(opt, feed_dict={input_ph: expert_data['observations'],
#                                     output_ph: np.expand_dims(expert_data['actions'], axis=1)})
#            if (train_step % 20 == 0):
#                step_rewards = []
#                print ("Step", train_step)
#                for episode in range(args.test_eps):
#                    epr = 0
#                    done = False
#                    obs = env.reset()
#                    while not done:
#                        action = sess.run(output_pred,
#                                 feed_dict={input_ph: np.expand_dims(obs, axis=0)})[0]
#                        obs, r, done, _ = env.step(action)
#                        epr += r
#                    step_rewards.append(epr)
#                reward_means.append(np.mean(step_rewards))
#                reward_stds.append(np.std(step_rewards))
#        data = {'means': np.array(reward_means),
#                'stds': np.array(reward_stds)}
#    with open(args.output_file, 'wb') as f:
#        pickle.dump(data, f)



