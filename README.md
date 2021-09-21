# AirSim-NH car with Soft Actor Critic
This is an implementation of AirSim-NH car using Soft Actor Critic.
I've been having lots of issues. It turns out the last bug has to do with 
dimensionalities inside actor, critic, and value networks.

It's working functionally, now. This means it is running. I just don't know
if it's learning. Perhaps it is.

The next on the todo is to record avg_score to determine if it is learning.
