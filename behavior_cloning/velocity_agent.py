import fluids
import pygame
import numpy as np
import torch
from velocity_cloning import BehavioralCloningModel

obs_args    = {"obs_dim":500, "shape":(80, 80)}   # **kwargs dictionary of arguments for observation construction
simulator = fluids.FluidSim(visualization_level=1,        # How much debug visualization you want to enable. Set to 0 for no vis
                            fps=0,                        # If set to non 0, caps the FPS. Target is 30
                            obs_space=fluids.OBS_GRID,# OBS_BIRDSEYE, OBS_GRID, or OBS_NONE
                            obs_args=obs_args,
                            background_control=fluids.BACKGROUND_CSP) # BACKGROUND_CSP or BACKGROUND_NULL

state = fluids.State(
    layout=fluids.STATE_CITY,
    background_cars=10,           # How many background cars
    controlled_cars=1,            # How many cars to control. Set to 0 for background cars only
    )

simulator.set_state(state)

car_keys = simulator.get_control_keys()

def get_velocity(obs, model_path="./models/bc_velocity_models/checkpoint_10000"):
    obs = 
    if model == None:
        H_SIZE = 20*20
        OUTPUT_SIZE = 1
        model = BehavioralCloningModel(obs_shape, act_shape)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        if IS_CUDA_ENABLED: model = model.cuda()


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

    #    actions = {k:fluids.VelocityAction(1) for k in car_keys}
    #    actions = {k:fluids.SteeringAction(0, 1) for k in car_keys}
    #    actions = {k:fluids.KeyboardAction() for k in car_keys}
    #    actions = {k:fluids.SteeringVelAction(0, 1) for k in car_keys}


    obs = simulator.get_observations(car_keys)
    raw_obs = obs.get_array()
    import ipdb; ipdb.set_trace()
    actions = get_velocity(raw_obs)
    rew = simulator.step(actions)
    simulator.render()
