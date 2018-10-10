import fluids
import datetime
import os
import sys
import time
import shutil


# Don't change this if you are working on AUTOLAB machine
data_root = "/nfs/diskstation/projects/fluids_dataset"


time_str = datetime.datetime.today().strftime('%Y-%m-%d')

# Fill out these fields
desc_str    = "behavior_cloning_obs_dim_500"   # Descriptive string for organizing saved data
n_cars      = 10   # Number of cars to collect observations over
n_peds      = 0   # Number of peds to collect observatoins over
car_lights  = False   # True/False place traffic lights
ped_lights  = False   # True/False place ped crossing lights
layout      = fluids.STATE_CITY   # Fluids Layout, fluids.STATE_CITY is one
obs_type    = fluids.OBS_GRID   # Observation type fluids.OBS_GRID or fluids.OBS_BIRDSEYE
obs_args    = {"obs_dim":500, "shape":(80, 80)}   # **kwargs dictionary of arguments for observation construction
action_type = fluids.VelocityAction   # Action type, fluids.VelocityAction, fluids.SteeringAccAction, etc.
batch_size  = 100
end_time    = 10000
make_dir    = True

# desc_str    = "test_dataset1"
# n_cars      = 10
# n_peds      = 0
# car_lights  = False
# ped_lights  = False
# layout      = fluids.STATE_CITY
# obs_type    = fluids.OBS_GRID
# obs_args    = {"obs_dim":500, "shape":(80, 80)}
# action_type = fluids.VelocityAction
# batch_size  = 10
# end_time    = 100
# make_dir    = True

if any([i == None for i in [desc_str, n_cars, n_peds, car_lights, ped_lights, layout, obs_type, obs_args, action_type, batch_size, end_time]]):
    print("Error, you didn't specify the data collection settings completely")
    exit(1)

if make_dir:
    print("WARNING. ARE YOU SURE YOU WANT TO DO THIS? YOU MIGHT BE OVERWRITING IMPORTANT DATA")
    print("Type \"yes\" to continue")
    a = input()
    assert(a == "yes")

folder_name = os.path.join(data_root, "{}_{}".format(time_str, desc_str))
try:
    os.mkdir(folder_name)
except FileExistsError:
    print("Error: {} already exists".format(folder_name))
    if make_dir: # Overwrite previous
        print("Overwriting {}".format(folder_name))
        shutil.rmtree(folder_name)
        os.mkdir(folder_name)
    else:
        exit(1)



simulator = fluids.FluidSim(visualization_level=0,
                            background_control=fluids.BACKGROUND_CSP)

state = fluids.State(layout=layout,
                     background_cars=n_cars,
                     background_peds=n_peds,
                     controlled_cars=0,
                     use_traffic_lights=car_lights,
                     use_ped_lights=ped_lights)
simulator.set_state(state)

data_saver = fluids.DataSaver(fluid_sim=simulator,
                              file=os.path.join(folder_name, "training_data"),
                              obs=[obs_type], #TODO: Make obs_args work
                              act=[action_type],
                              batch_size=batch_size,
                              obs_kwargs=obs_args)

simulator.set_data_saver(data_saver)

t = 0
curr = time.time()
while not end_time or t < end_time:
    print(t)
    simulator.step()
    simulator.render()
    if simulator.in_deadlock():
        state = fluids.State(layout=layout,
                     background_cars=n_cars,
                     background_peds=n_peds,
                     controlled_cars=0,
                     use_traffic_lights=car_lights,
                     use_ped_lights=ped_lights)
        simulator.set_state(state)

    t = t + 1

