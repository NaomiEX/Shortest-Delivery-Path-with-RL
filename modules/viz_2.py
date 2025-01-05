def plot_training_loss(Engine):
    training_loss = Engine.get_training_loss()
    y = [x for x in training_loss if x != None]

    plt.plot([i+1 for i in range(len(y))], y)
    plt.title('Training Loss')
    plt.xlabel('Batch #')
    plt.ylabel('Loss')
    plt.show()

plot_training_loss(Engine)


import matplotlib.pyplot as plt

def plot_q_values_3d(learning):
    """
        Plot the summation of Q-values of the actions (from adjacent cells) that leads to cells in 3D heat map. 

    Args:
        learning (Learning): The learning algorithm of the problem.
    """
    table = learning.table

    x, y = GRID_SIZE

    movement_list = [Movement.idx_to_cls(i) for i in range(4)]
    down_index = movement_list.index(MoveDown)
    right_index = movement_list.index(MoveRight)
    up_index = movement_list.index(MoveUp)
    left_index = movement_list.index(MoveLeft)

    # learning.table(state) -> x   x[0] -> q_value   Movement.idx_to_cls(0) see which action

    # Create empty grid
    grid_plot = np.zeros((2*x -1, 2*y - 1))

    print(grid_plot)

    # Find the sum of Q-values
    for i in range(-x+1, x):
        for j in range(-y+1, y):
            
            grid_pos_x = i+x-1
            grid_pos_y = j+y-1

            # Q-value of action to MoveDown from cell above to current cell 
            if j - 1 > -y:
                grid_plot[grid_pos_x][grid_pos_y] += table([i, j-1]).squeeze()[down_index]

            # Q-value of action to MoveRight from cell at the left of current cell 
            if i - 1 > -x:
                grid_plot[grid_pos_x][grid_pos_y] += table([i-1, j]).squeeze()[right_index]

            # Q-value of action to MoveUp from cell below to current cell 
            if j + 1 < y:
                grid_plot[grid_pos_x][grid_pos_y] +=  table([i, j+1]).squeeze()[up_index]

            # Q-value of action to MoveLeft from cell at the right of current cell 
            if i + 1 < x:
                grid_plot[grid_pos_x][grid_pos_y] += table([i+1, j]).squeeze()[left_index]

    state_1, state_2 = np.meshgrid(np.arange(2*x -1)-x, np.arange(2*y -1)-y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    surf = ax.plot_surface(state_1, state_2, grid_plot, cmap='viridis')

    ax.set_xlabel('Row')
    ax.set_ylabel('Column')
    fig.colorbar(surf, ax=ax, shrink = 0.5, aspect=10, label='Q-Value Summation')

    plt.title( "3-D Heat Map: Summation of Q-Values of Actions which Leads to The State" )
    plt.show()

plot_q_values_3d(learning)




def plot_q_values_2d(learning):
    """
        Plot the summation of Q-values of the actions (from adjacent cells) that leads to cells in 3D heat map. 

    Args:
        learning (Learning): The learning algorithm of the problem.
    """
    table = learning.table

    x, y = GRID_SIZE

    movement_list = [Movement.idx_to_cls(i) for i in range(4)]
    down_index = movement_list.index(MoveDown)
    right_index = movement_list.index(MoveRight)
    up_index = movement_list.index(MoveUp)
    left_index = movement_list.index(MoveLeft)

    # learning.table(state) -> x   x[0] -> q_value   Movement.idx_to_cls(0) see which action

    # Create empty grid
    grid_plot = np.zeros((2*x -1, 2*y - 1))

    print(grid_plot)

    # Find the sum of Q-values
    for i in range(-x+1, x):
        for j in range(-y+1, y):
            
            grid_pos_x = i+x-1
            grid_pos_y = j+y-1

            # Q-value of action to MoveDown from cell above to current cell 
            if j - 1 > -y:
                grid_plot[grid_pos_x][grid_pos_y] += table([i, j-1]).squeeze()[down_index]

            # Q-value of action to MoveRight from cell at the left of current cell 
            if i - 1 > -x:
                grid_plot[grid_pos_x][grid_pos_y] += table([i-1, j]).squeeze()[right_index]

            # Q-value of action to MoveUp from cell below to current cell 
            if j + 1 < y:
                grid_plot[grid_pos_x][grid_pos_y] +=  table([i, j+1]).squeeze()[up_index]

            # Q-value of action to MoveLeft from cell at the right of current cell 
            if i + 1 < x:
                grid_plot[grid_pos_x][grid_pos_y] += table([i+1, j]).squeeze()[left_index]

    plt.imshow( grid_plot )
    plt.xticks([x for x in range(2*x-1)], [x for x in range(-x+1, x)])
    plt.yticks([y for y in range(2*y-1)], [y for y in range(-y+1, y)])
    plt.xlabel('Row')
    plt.ylabel('Column')
    plt.title( "2-D Heat Map: Summation of Q-Values of Actions which Leads to the State" )
    plt.colorbar(label='Q-Values Summation')

    plt.show()

plot_q_values_2d(learning)

