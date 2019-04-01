import numpy as np
import tensorflow as tf
import gym

env = gym.make('Breakout-v0')
tf.enable_eager_execution()

def preprocess(img):
    img = tf.convert_to_tensor(img)
    img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, (100, 100))
    img = tf.image.central_crop(img, 0.85)
    
    return img

learning_rate = 1e-3
gamma = .99

# Constants defining our neural network
input_size = (86, 86, 1)
output_size = env.action_space.n

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu', input_shape=input_size),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2,2), activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2,2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax'),
])

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


def get_action(xs):
    xs = tf.convert_to_tensor(xs)
    action_pred = model(xs)
    
    return action_pred.numpy()[0]

def calc_loss(xs, ys, advantages):
    xs = tf.convert_to_tensor(xs)
    ys = tf.convert_to_tensor(ys)
    advantages = tf.convert_to_tensor(advantages)

    with tf.GradientTape() as tape:
        action_pred = model(xs, training=True)
        log_lik = -ys * tf.log(action_pred)
        log_lik_adv = log_lik * advantages
        loss_value = tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis=1))
        
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables),
                            global_step=tf.train.get_or_create_global_step())
    

def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    return discounted_r

num_episodes = 1000
# This list will contain episode rewards from the most recent 100 games
# Clear Condition: Average reward per episode >= 195.0 over 100 games
EPISODE_100_REWARD_LIST = []

for i in range(num_episodes):

    # Clear out game variables
    xs = np.empty(shape=[0, 86, 86, 1])
    ys = np.empty(shape=[0, output_size])
    rewards = np.empty(shape=[0, 1])

    reward_sum = 0
    state = preprocess(env.reset())

    while True:
        # Append the observations to our batch
        x = np.reshape(state, [1, 86, 86, 1]).astype(np.float32)
        
        # Run the neural net to determine output
        action_prob = get_action(x)
        action = np.random.choice(np.arange(output_size), p=action_prob)
    
        # Append the observations and outputs for learning
        xs = np.vstack([xs, x])
        y = np.zeros(output_size)
        y[action] = 1
        
        ys = np.vstack([ys, y])

        # Determine the outcome of our action
        env.render()
        state, reward, done, _ = env.step(action)
        state = preprocess(state)
        reward_sum += reward
        rewards = np.vstack([rewards, reward])
        
        if done:
            discounted_rewards = discount_rewards(rewards, gamma)
            # Normalization
            discounted_rewards = (discounted_rewards - discounted_rewards.mean())/(discounted_rewards.std() + 1e-7)
            
            xs, ys = xs.astype(np.float32), ys.astype(np.float32)
            print(xs.shape)
            calc_loss(xs, ys, discounted_rewards)
            break
    
    print(i, reward_sum)
