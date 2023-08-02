import matplotlib.pyplot as plt



def plot_statistics(prin_rewards, prin_chosen_arms, prin_regrets, prin_optimalities, parameter, name):

    plt.subplot(4, 1, 1)
    for i, reward in enumerate(prin_rewards):
        plt.plot(range(len(reward)), reward,
                    label=f"reward {i}, {name} {parameter}")
        plt.legend()

    plt.subplot(4, 1, 2)
    for i, action in enumerate(prin_chosen_arms):
        plt.plot(range(len(action)), action,
                    label=f"action sequence {i}, {name} {parameter}")
        plt.legend()

    plt.subplot(4, 1, 3)
    for i, action in enumerate(prin_regrets):
        plt.plot(range(len(action)), action,
                    label=f"regrets {i}, {name} {parameter}")
        plt.legend()

    plt.subplot(4, 1, 4)
    for i, action in enumerate(prin_optimalities):
        plt.plot(range(len(action)), action,
                    label=f"optimalities {i}, {name} {parameter}")
        plt.legend()

    plt.show()