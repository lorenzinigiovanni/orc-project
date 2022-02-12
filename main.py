import time
import random
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F

from network import Network
from mypendulum import MyPendulum
import tables


def main():
    # --- Random seed ---
    RANDOM_SEED = int((time.time() % 10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Hyper parameters ---
    # Learning rate of the neural network
    QVALUE_LEARNING_RATE = 1e-3
    # Update Q_targets once every steps
    UPDATE_STEPS = 100
    # Update the model once every CYCLE
    CYCLE = 4
    # Minimum number of steps to train the model
    MIN_STEPS = 1024
    # Memory of the replay buffer
    MEMORY = 1000000
    # Size of the batch
    BATCH_SIZE = 64

    # Number of training episodes
    N_EPISODES = 500
    # Lenght of a training episode
    MAX_EPISODE_LENGTH = 250
    # Discount factor
    DISCOUNT = 0.9
    # Initialize the exploration probability to 1
    EXPLORATION_PROB = 1
    # Exploration decay
    EXPLORATION_DECAY = 0.01
    # Minimum of exploration probability
    MIN_EXPLORATION_PROB = 0.001

    # Train the network
    TRAINING = True
    # Show a run using the weights learned by the network
    TESTING = True
    # Continue training the network from the saved weights
    CONTINUE_TRAINING = False
    # Print policy and value tables
    PRINT_TABLES = True

    # --- Environment ---
    # Number of joints of the robot
    N_JOINTS = 1
    # Number of discretization steps for the joint torque u
    NU = 11
    # Number of joint position + number of joint velocity
    NX = 2 * N_JOINTS

    # Create a pendulum
    env = MyPendulum(N_JOINTS, NU)

    # --- Neural networks ---
    # Critic NN
    Q = Network(NX, NU, N_JOINTS)
    # Target NN
    Q_target = Network(NX, NU, N_JOINTS)

    # Load the model
    # Note: replay memory won't be loaded
    if(CONTINUE_TRAINING):
        # Draw the weights from the last session
        with open('Q.pt', 'rb') as f:
            Q.load_state_dict(torch.load(f))

    # Set initial weights of targets equal to those of actor and critic
    Q_target.load_state_dict(Q.state_dict())

    # --- Training ---
    if(TRAINING):
        # Set optimizer specifying the learning rates
        critic_optimizer = torch.optim.Adam(
            Q.parameters(),
            lr=QVALUE_LEARNING_RATE,
        )

        # Replay buffer
        # Store states
        x_r = torch.zeros(MEMORY, NX)
        # Store actions
        u_r = torch.zeros(MEMORY, N_JOINTS, 1)
        # Store costs
        c_r = torch.zeros(MEMORY, 1)
        # Store next states
        x_next_r = torch.zeros(MEMORY, NX)
        # # Store done flags
        # done_r = torch.zeros(MEMORY, 1)
        # Index of the current transition in the replay buffer
        i_r = 0
        # Count the number of transitions in the replay buffer
        n_r = 0

        # # Store the best cost to go
        # best_cost_to_go = np.inf
        # Count the number of steps
        step = 0

        try:
            # For each episode
            for e in range(N_EPISODES):
                # Get a random initial state
                env.reset()

                # Set the network's mode to train: allows gradient updates
                Q.train()

                # Simulate an episode up to MAX_EPISODE_LENGTH's steps
                for t in range(MAX_EPISODE_LENGTH):
                    # Increment the counter of steps
                    step += 1

                    # Source state from environment
                    x = env.x
                    # Convert the state to a tensor with dimension [1, NX]
                    x = torch.tensor(x.astype(np.float32)).unsqueeze(0)

                    # Take a random action u with probability < exploration prob
                    if np.random.rand() < EXPLORATION_PROB:
                        # u is a random action with dimension [1, N_JOINTS]
                        u = torch.tensor([np.random.randint(0, NU) for _ in range(N_JOINTS)])

                    # Use an action dictated by the Q network
                    else:
                        # u is the best action with dimension [1, N_JOINTS]
                        u = Q(x).min(2).indices.squeeze(0)

                    # Do a step in the environment
                    x_next, cost = env.step(np.array(u))
                    # x_next is a tensor with dimension [1, NX]
                    x_next = torch.tensor(x_next.astype(np.float32))

                    # Save current elements in replay buffer
                    x_r[i_r] = x
                    u_r[i_r] = u.unsqueeze(1)
                    c_r[i_r] = cost
                    x_next_r[i_r] = x_next
                    # done_r[i_r] = 1 if t == MAX_EPISODE_LENGTH - 1 else 0
                    n_r += 1
                    i_r += 1
                    # Keep i_r between 0 and MEMORY - 1 so to make a circular buffer
                    i_r = i_r % MEMORY

                    # Every few steps, train the network
                    if(n_r > MIN_STEPS and step % CYCLE == 0):
                        # Sample a batch of transitions from the replay buffer
                        indexes = random.choices(range(min(MEMORY, n_r)), k=BATCH_SIZE)

                        # Load the batch of transitions
                        x = x_r[indexes]
                        u = u_r[indexes]
                        c = c_r[indexes]
                        x_next = x_next_r[indexes]
                        # done = done_r[indexes]

                        # Set the gradients to zero
                        critic_optimizer.zero_grad()

                        # Q-Learning target is Q(x, u) <- c + γ min_u Q_t(x', u')
                        # target is a tensor with dimension [BATCH_SIZE, N_JOINTS]
                        # target = c.broadcast_to((c.shape[0], N_JOINTS)) + torch.mul(DISCOUNT * Q_target(x_next).min(2).values, 1 - done)
                        target = c.broadcast_to((c.shape[0], N_JOINTS)) + DISCOUNT * Q_target(x_next).min(2).values

                        # Q-Learning current is min_u Q(x, u)
                        # current is a tensor with dimension [BATCH_SIZE, N_JOINTS]
                        current = Q(x).gather(2, u.long()).squeeze(2)

                        # Get the loss between current and target: loss is squared around 0, linear elsewhere
                        loss = F.smooth_l1_loss(current, target)

                        # Backpropagate the loss
                        loss.backward()

                        # Update the weights
                        critic_optimizer.step()

                    # Update Q_target once every UPDATE_STEPS step
                    if(step % UPDATE_STEPS == 0):
                        Q_target.load_state_dict(Q.state_dict())

                # Update EXPLORATION_PROB depending on epoch and EXPLORATION_DECAY
                EXPLORATION_PROB = max(MIN_EXPLORATION_PROB, np.exp(-EXPLORATION_DECAY*e))

                # # Test the Q network with a fixed starting point

                # # q0 is the initial position with dimension [1, N_JOINTS]
                # q0 = np.zeros(N_JOINTS)
                # q0[0] = np.pi

                # # v0 is the initial velocity with dimension [1, N_JOINTS]
                # v0 = np.zeros(N_JOINTS)

                # # x0 is the initial state with dimension [2, N_JOINTS]
                # x0 = np.vstack([q0, v0])

                # # Reset the environment with state x0
                # env.reset(x0)

                # # Initialize the cost to go and gamma
                # cost_to_go = 0.0
                # gamma = 1.0

                # # Set the NN in evaluation mode
                # Q.eval()

                # # Do a run, getting the cost and evaluating the NN
                # for t in range(MAX_EPISODE_LENGTH):
                #     # Source state from environment
                #     x = env.x
                #     # Convert the state to a tensor with dimension [1, NX]
                #     x = torch.tensor(x.astype(np.float32)).unsqueeze(0)

                #     # u is the best action with dimension [1, N_JOINTS]
                #     u = Q(x).min(2).indices.squeeze(0)

                #     # Do a step in the environment
                #     x_next, cost = env.step(np.array(u))

                #     # Track the cost to go
                #     cost_to_go += cost * gamma
                #     # Update gamma by discounting it
                #     gamma *= DISCOUNT

                # Debug print
                print("End of epoch:", e) # , " Cost:", cost_to_go)

                # # If a model reaching a better cost to go is found, then update the current best cost
                # if(cost_to_go < best_cost_to_go):
                #     best_cost_to_go = cost_to_go

                #     # Save the best model weights
                #     with open('Q.pt', 'wb') as f:
                #         torch.save(Q.state_dict(), f)

        # Interrupt the training if CTRL + C is pressed
        except KeyboardInterrupt:
            with open('Q.pt', 'wb') as f:
                torch.save(Q.state_dict(), f)

        with open('Q.pt', 'wb') as f:
            torch.save(Q.state_dict(), f)

    # --- Tables ---
    if(PRINT_TABLES):
        # Load the best model weights
        with open('Q.pt', 'rb') as f:
            Q.load_state_dict(torch.load(f))

        # Set the NN in evaluation mode
        Q.eval()

        # Store the policy in a table
        P = np.zeros((tables.NQ, tables.NV))
        # Store the values in a table
        V = np.zeros((tables.NQ, tables.NV))

        # For every possible state x
        for iq in range(tables.NQ):
            for iv in range(tables.NV):
                # Convert the state to a tensor with dimension [1, NX]
                x = np.array([tables.d2cq(iq), tables.d2cv(iv)])
                x = torch.tensor(x.astype(np.float32)).unsqueeze(0)

                # Call the network Q to get Q
                Q_val = Q(x)

                # Save the best index in the policy table
                P[iq][iv] = Q_val.detach().min(2).indices.squeeze(0).numpy()
                # Save the best value in the value table
                V[iq][iv] = Q_val.detach().min(2).values.squeeze(0).numpy()

        tables.plot_policy(P.T)
        tables.plot_V_table(V.T)

    # --- Testing ---
    if(TESTING):
        # Load the best model weights
        with open('Q.pt', 'rb') as f:
            Q.load_state_dict(torch.load(f))

        # Test the Q network with a fixed starting point

        # q0 is the initial position with dimension [1, N_JOINTS]
        q0 = np.zeros(N_JOINTS)
        q0[0] = np.pi

        # v0 is the initial velocity with dimension [1, N_JOINTS]
        v0 = np.zeros(N_JOINTS)

        # x0 is the initial state with dimension [2, N_JOINTS]
        x0 = np.vstack([q0, v0])

        # Reset the environment with state x0
        env.reset(x0)

        # Set the NN in evaluation mode
        Q.eval()

        # Save state joint angles and velocities:
        pos = []
        # Save controls u
        u_hist = []

        # Simulate an episode up to MAX_EPISODE_LENGTH's steps
        for t in range(MAX_EPISODE_LENGTH):
            # Source state from environment
            x = env.x
            # Update state history
            pos.append(x.copy())
            # Convert the state to a tensor with dimension [1, NX]
            x = torch.tensor(x.astype(np.float32)).unsqueeze(0)

            # u is the best action with dimension [1, N_JOINTS]
            u = Q(x).min(2).indices.squeeze(0)

            # Update control history
            u_hist.append(u)

            # Do a step in the environment
            x_next, cost = env.step(np.array(u))

            # Show in Gepetto
            env.render()

        # Axis x for the plot
        lenghts = [t for t in range(MAX_EPISODE_LENGTH)]

        # Transpose pos and u_hist.
        # This way you can access to data by accessing rows instead of cycling every column in the lists
        pos = list(map(list, zip(*pos)))
        u_hist = list(map(list, zip(*u_hist)))

        # Stack two subplots vertically
        _, (ax1, ax2) = plt.subplots(2)
        
        for i in range(0, N_JOINTS):
            # Plot the joint angles
            ax1.plot(lenghts, pos[i], label='Position joint ' + str(i))
            # Plot the joint velocities
            ax2.plot(lenghts, u_hist[i], label='Action joint ' + str(i))

        ax1.set_title('Joint positions')
        ax1.legend()

        ax2.set_title('Joint actions')
        ax2.legend()

        plt.show()


main()
