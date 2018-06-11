import gym
import gym_urbandriving as uds
import time
import numpy as np
import json

config = json.load(open('configs/sensitivity_config.json'))

config['environment']['visualize'] = False

n_trials = 20

while (True):
    f = open("sensitivity_out.txt", "a+")
    config['agents']['background_cars'] = np.random.randint(2, 10)
    config['agents']['use_traffic_lights'] = np.random.random() < 0.5
    config['agents']['use_pedestrians'] = True
    config['agents']['number_of_pedestrians'] = np.random.randint(0, 6)
    config['agents']['bg_state_space_config']['noise'] = np.random.random()
    config['agents']['bg_state_space_config']['omission_prob'] = np.random.random()
    print(config)
    env = uds.UrbanDrivingEnv(config_data=config, randomize=True)
    
    env._reset()
    env._render()
    z = 0
    n_successes = 0
    while (True):
        if (z == n_trials):
            break
        state, _, done, _ = env._step([])
        env._render()
        state = env.current_state
        done = False
        success = True

        # for k in state.dynamic_objects['background_cars']:
        #     car = state.dynamic_objects['background_cars'][k]
        #     if car.trajectory.npoints():
        #         success = False
        for i in range(config['agents']['background_cars']):
            if state.collides_any_dynamic(i):
                done = True
                success = False
            elif state.time > config['environment']['max_time']:
                done = True
                success = True

        if (done):
            print(success)
            if success:
                n_successes += 1
            env._reset()
            z += 1
    print(n_successes / n_trials)
    f.write(str(config['agents']['background_cars']) + ",")
    f.write(str(int(config['agents']['use_traffic_lights'])) + ",")
    f.write(str(int(config['agents']['use_pedestrians'])) + ",")
    f.write(str(config['agents']['number_of_pedestrians']) + ",")
    f.write(str(config['agents']['bg_state_space_config']['noise']) + ",")
    f.write(str(config['agents']['bg_state_space_config']['omission_prob']) + ",")
    f.write(str(n_successes / n_trials) + '\n')
    f.close()
