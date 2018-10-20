import fluids
import glob
import sklearn
import numpy as np
import tqdm

import torch
from torch import nn
import torch.nn.functional as F

"""
Basic IL agent - discrete classification model for supervised data
Grid Observations -> Velocity Actions
# sklearn:
    # 1. neural networks
    # 2. decision trees
    # 3. SVM
    # 4. k nearest neighbors
    # 5. linear/logistic regression
    # 6. probabilistic / Bayes modelling
should be solvable (no traffic, pedestrian) 
downsample low resolution GridObservation class, only select used channels
"""

# TODO: change these to regular demonstrations
DATA_ROOT = "../../fluids_data/2018-10-07_behavior_cloning/"
MODEL_ROOT = "../../fluids_data/models/bc_velocity_models"

CHANNELS = [
    "terrain_window",
    "drivable_window",
    "undrivable_window",
    "car_window",
    "ped_window",
    "light_window_red",
    "light_window_green",
    "light_window_yellow",
    "direction_window",
    "direction_pixel_window",
    "direction_edge_window"
]
NUM_CHANNELS = "num_channels"
IS_CUDA_ENABLED = torch.cuda.is_available()
MODES = {
    "NN": {
        "convs": [ (NUM_CHANNELS, 4), (1, 6)],
        "fcs": [100],
        "activation": F.relu
    },
}
USING_CHANNELS = [0, 1, 2, 3, -3]

def data_generator(data_path):
    arr = np.load(data_path)["arr_0"]
    np.random.shuffle(arr)
    for car in arr:
        print(car.shape)
        obs, act = car[0][2], car[0][3]
        obs = obs[:, :, USING_CHANNELS]
        obs = np.moveaxis(obs, 2, 0)
        obs, act = np.expand_dims(obs, 0), np.expand_dims(act, 0)
        obs = torch.from_numpy(obs)
        act = torch.from_numpy(act)
        yield obs.type('torch.FloatTensor'), act.type('torch.FloatTensor')

class ILAgent(nn.Module):
    # TODO: what numbers to pick for conv layers?
    def __init__(self, obs_shape, act_shape, mode = "NN"):
        super(ILAgent, self).__init__()
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.mode = mode
        num_channels = obs_shape[1] # (None, num_channels, height, width) for grid
        curr_channels = num_channels
        
        if self.mode == "NN":
            convs = MODES["NN"]["convs"]
            fcs = MODES["NN"]["fcs"]
            for i, conv in enumerate(convs):
                if conv[0] == NUM_CHANNELS:
                    conv = (num_channels, conv[1])
                convs[i] = nn.Conv2d(curr_channels, conv[0], conv[1])
                curr_channels = conv[0]

            self.convs = nn.ModuleList(convs)
            self.conv_final_size = self.calc_conv_final_size

            dim_i = self.conv_final_size
            for i, dim_o in enumerate(fcs):
                fcs[i] = nn.Linear(dim_i, dim_o)
                dim_i = dim_o
            fcs.append(nn.Linear(dim_i, act_shape[1]))
            self.fcs = nn.ModuleList(fcs)

            self.activation = MODES["NN"]["activation"]

    def calc_conv_final_size(self):
        sample = torch.randn(self.obs_shape)
        for conv in self.convs:
            sample = conv(sample)
        size = sample.view(sample.size(0), -1).size(1)
        return size

    def forward(self, x):
        y = x
        if self.mode == "NN":
            for conv in self.convs:
                y = self.activation(conv(y))
            # y = y.view(y.size(0), -1)
            for fc in self.fcs:
                y = F.relu(fc(y))
            return y

def train(mode):
    loss_fn = nn.MSELoss(reduction="sum")
    learning_rate = 1e-6
    model = None
    min_loss = float('inf')
    best_model_params = None
    PRINT_LOSS_ITERS = 100
    SAVE_MODEL_ITERS = 500
    total_loss = 0

    for data_path in glob.glob("{}/*.npz".format(DATA_ROOT)):
        for i, (obs, act) in tqdm(enumerate(data_generator(data_path))):
            if model == None:
                model = ILAgent(obs.shape, act.shape, mode=mode)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                if IS_CUDA_ENABLED: 
                    model = model.cuda()
                
                act_pred = model(obs)
                loss = loss_fn(act_pred, act)
                total_loss += loss.item()

                if i % PRINT_LOSS_ITERS == 0:
                    print("Iteration {} loss: {}".format(i, total_loss / i))
                    total_loss = 0
                if i % SAVE_MODEL_ITERS == 0:
                    torch.save(model.sate_dict(), "{}/checkpoint_{}".format(MODEL_ROOT, i))
                if loss < min_loss:
                    min_loss = loss
                    best_model_params = model.state_dict()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    print("Min loss: {}".format(min_loss))
    torch.save(best_model_params, "{}/best_model".format(MODEL_ROOT))

def test_agent(mode):
    obs_args = {"obs_dim": 500, "shape": (80, 80)}
    simulator = fluids.FluidSim(visualization_level=1,
                                fps=0,
                                obs_space=fluids.OBS_GRID,
                                obs_args=obs_args,
                                background_control=fluids.BACKGROUND_CSP)
    state = fluids.State(
        layout=fluids.STATE_CITY,
        background_cars=18,
        controlled_cars=1
    )

    simulator.set_state(state)
    car_keys = simulator.get_control_keys()

    model = None

    def get_velocity(obs, model_path="{}/checkpoint_90000".format(MODEL_ROOT)):
        nonlocal model
        # TODO: doesn't this just terminate after 1 iteration???
        if model == None:
            for i, (obs_, act_) in enumerate(data_generator("{}/training_data_0.npz".format(DATA_ROOT))):
                model = ILAgent(obs_.shape, act_.shape, mode=mode)
                if IS_CUDA_ENABLED:
                    model = model.cuda()
                    model.load_state_dict(torch.load(model_path))
                else:
                    model.load(state_dict(torch.load(model_path, map_location='cpu')))
                break
        obs = np.moveaxis(obs, 2, 0)
        obs = np.expand_dims(obs, 0)
        obs = torch.from_numpy(obs)
        obs = obs.type('torch.FloatTensor')
        act = model(obs).detach().numpy()[0][0]
        return act

    while True:
        # TODO: what is this used for?
        actions_s = simulator.get_supervisor_actions(fluids.VelocityAction, keys=car_keys)

        obs = simulator.get_observations(car_keys)
        raw_obs = [obs[k] for k in obs][0].get_array()[:, :, USING_CHANNELS]
        velocity = get_velocity(raw_obs)
        print("Velocity: {}".format(velocity))
        actions = {k: fluids.VelocityAction(velocity) for k in car_keys}
        rew = simulator.step(actions)
        simulator.render()

def main():
    # TODO: loop over epochs
    train(mode="NN")

if __name__ == "__main__":
    main()
        



