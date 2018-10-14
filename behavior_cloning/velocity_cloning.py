import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import glob
import pygame
import fluids

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
USING_CHANNELS = [0, 1, 2, 3, -3]

def single_data_generator(data_path):
    #yields grid_obs, action
    arr = np.load(data_path)["arr_0"]
    np.random.shuffle(arr)
    for car in arr:
        #print(car.shape)
        obs, act = car[0][2], car[0][3]
        obs = obs[:, :, USING_CHANNELS]
        obs = np.moveaxis(obs, 2, 0)
        obs, act = np.expand_dims(obs, 0), np.expand_dims(act, 0)
        obs = V(torch.from_numpy(obs))
        act = V(torch.from_numpy(act))
        yield obs.type('torch.FloatTensor'), act.type('torch.FloatTensor')



class BehavioralCloningModel(torch.nn.Module):
    def __init__(self, obs_shape, act_shape, convs=[(NUM_CHANNELS, 4), (1, 6)], fcs=[400]):
        super(BehavioralCloningModel, self).__init__()
        #import ipdb; ipdb.set_trace()
        num_channels = obs_shape[1] # (None, num_channels, height, width) for grid
        curr_channels = num_channels
        for i, conv in enumerate(convs):
            if conv[0] == NUM_CHANNELS:
                conv = (num_channels, conv[1])
            convs[i] = torch.nn.Conv2d(curr_channels, conv[0], conv[1])
            curr_channels = conv[0]

        self.convs = torch.nn.ModuleList(convs)
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.conv_final_size = self.calc_conv_final_size()

        dim1 = self.conv_final_size
        for i, dim2 in enumerate(fcs):
            fcs[i] = torch.nn.Linear(dim1, dim2)
            dim1 = dim2
        fcs.append(torch.nn.Linear(dim1, act_shape[1]))
        self.fcs = torch.nn.ModuleList(fcs)


    def calc_conv_final_size(self):
        sample_x = V(torch.randn(self.obs_shape))
        for conv in self.convs:
            sample_x = conv(sample_x)
        #import ipdb; ipdb.set_trace()
        size = sample_x.view(sample_x.size(0), -1).size(1)
        return size

    
    def forward(self, x):
        #import ipdb; ipdb.set_trace()
        y = x
        for conv in self.convs:
            y = F.relu(conv(y))
        y = y.view(y.size(0), -1)
        for fc in self.fcs:
            y = F.relu(fc(y))
        return y
            

# TODO: add epochs, add lr_find, add data_loader class

def train():
    loss_fn = torch.nn.MSELoss(reduction='sum')
    learning_rate = 1e-7
    model = None
    min_loss = float('inf')
    best_model_params = None
    PRINT_LOSS_ITERS = 100
    SAVE_MODEL_ITERS = 500
    total_loss = 0
    for data_path in glob.glob("{}/*.npz".format(DATA_ROOT)):
        for i, (obs, act) in tqdm(enumerate(single_data_generator(data_path))):
            if model == None:
                H_SIZE = 20*20
                OUTPUT_SIZE = 1
                model = BehavioralCloningModel(obs.shape, act.shape)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                if IS_CUDA_ENABLED: model = model.cuda()

            # Forward pass: compute predicted y by passing x to the model.
            act_pred = model(obs)

            # Compute and print loss.
            loss = loss_fn(act_pred, act)
            total_loss += loss.item()
            if i % PRINT_LOSS_ITERS == 0: 
                print(i, total_loss / PRINT_LOSS_ITERS)
                total_loss = 0
            if i % SAVE_MODEL_ITERS == 0:
                torch.save(model.state_dict(), "{}/checkpoint_{}".format(MODEL_ROOT, i))
            if loss < min_loss:
                min_loss = loss
                best_model_params = model.state_dict()

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()


    print("Min loss was:", min_loss)
    torch.save(best_model_params, "{}/best_model".format(MODEL_ROOT))

def test_agent():

    obs_args    = {"obs_dim":500, "shape":(80, 80)}   # **kwargs dictionary of arguments for observation construction
    simulator = fluids.FluidSim(visualization_level=1,        # How much debug visualization you want to enable. Set to 0 for no vis
                                fps=0,                        # If set to non 0, caps the FPS. Target is 30
                                obs_space=fluids.OBS_GRID,# OBS_BIRDSEYE, OBS_GRID, or OBS_NONE
                                obs_args=obs_args,
                                background_control=fluids.BACKGROUND_CSP) # BACKGROUND_CSP or BACKGROUND_NULL

    state = fluids.State(
        layout=fluids.STATE_CITY,
        background_cars=18,           # How many background cars
        controlled_cars=1,            # How many cars to control. Set to 0 for background cars only
        )

    simulator.set_state(state)

    car_keys = simulator.get_control_keys()

    model = None
    def get_velocity(obs, model_path="{}/checkpoint_90000".format(MODEL_ROOT)):
        nonlocal model
        if model == None:
            for i, (obs_, act_) in enumerate(single_data_generator("{}/training_data_0.npz".format(DATA_ROOT))):
                H_SIZE = 20*20
                OUTPUT_SIZE = 1
                model = BehavioralCloningModel(obs_.shape, act_.shape)
                if IS_CUDA_ENABLED: 
                    model = model.cuda()
                    model.load_state_dict(torch.load(model_path))
                else:
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                break
        obs = np.moveaxis(obs, 2, 0)
        obs = np.expand_dims(obs, 0)
        obs = V(torch.from_numpy(obs))
        obs = obs.type('torch.FloatTensor')
        act = model(obs).detach().numpy()[0][0]
        return act

    while True:
        actions = {}

        # Uncomment any of these lines.
        # VelocityAction is vel for car to move along trajectory
        # SteeringAction is steer, acc control
        # KeyboardAction is use keyboard input
        # SteeringVelAction is steer, vel control


        # actions = simulator.get_supervisor_actions(fluids.SteeringAction, keys=car_keys)
        # actions = simulator.get_supervisor_actions(fluids.VelocityAction, keys=car_keys)
        # actions = simulator.get_supervisor_actions(fluids.SteeringAccAction, keys=car_keys)

        #    actions = {k:fluids.SteeringAction(0, 1) for k in car_keys}
        #    actions = {k:fluids.KeyboardAction() for k in car_keys}
        #    actions = {k:fluids.SteeringVelAction(0, 1) for k in car_keys}


        obs = simulator.get_observations(car_keys)
        raw_obs = [obs[k] for k in obs][0].get_array()[:, :, USING_CHANNELS]
        vel = get_velocity(raw_obs)
        print(vel)
        actions = {k:fluids.VelocityAction(vel) for k in car_keys}
        rew = simulator.step(actions)
        simulator.render()
