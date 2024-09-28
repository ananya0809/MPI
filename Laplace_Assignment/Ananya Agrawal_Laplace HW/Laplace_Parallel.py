from mpi4py import MPI
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as cols

# Constants for the metal plate matrix representation
ROWS, COLUMNS = 1000, 1000  # Grid size
MAX_TEMP_ERROR = 0.01  # Convergence criteria
MAX_ITERATIONS = 4000  # Maximum iterations

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Determine the number of rows each process handles
local_rows = ROWS // size


def initialize_source_plate(temp):
    temp[:, :] = 0
    #Set right side boundary condition
    for i in range(ROWS + 1):
        temp[i, COLUMNS + 1] = 100 * np.sin(((3.14159 / 2) / ROWS) * i)
    # Set bottom boundary condition
    for i in range(COLUMNS + 1):
        temp[ROWS + 1, i] = 100 * np.sin(((3.14159 / 2) / COLUMNS) * i)


source_temp = np.zeros((ROWS + 2, COLUMNS + 2))
initialize_source_plate(source_temp)


# Function to initialize the temperature grid
def initialize_temperature(temp, rows, cols, cur_rank):
    temp[:, :] = 0  # Set everything to 0 initially

    # Boundary condition on the right side
    start_row = cur_rank * local_rows
    for i in range(rows):
        temp[i, cols - 1] = source_temp[start_row + i, cols - 1]

    # Boundary condition on the bottom side
    for j in range(cols):
        temp[rows - 1, j] = 100 * np.sin(((3.14159 / 2) / COLUMNS) * j)


temperature = np.zeros((local_rows + 2, COLUMNS + 2))  # Extra rows for ghost cells
# Allocate local grids with ghost cells
temperature_last = np.zeros_like(temperature)

# Initialize the temperature grid
initialize_temperature(temperature_last, local_rows + 2, COLUMNS + 2, rank)

# Main Laplace loop
dt = MAX_TEMP_ERROR + 1
iteration = 0

while dt > MAX_TEMP_ERROR and iteration < MAX_ITERATIONS:
    # Exchange boundary rows with neighboring processes (ghost cells)
    if rank < size - 1: # PE 0, 1, 2
        comm.Send(temperature_last[-2, :], dest=rank + 1)  # Send bottom row
        comm.Recv(temperature_last[-1, :], source=rank + 1)  # Receive from below
    if rank > 0: # PE 1, 2, 3
        comm.Recv(temperature_last[0, :], source=rank - 1)  # Receive from above
        comm.Send(temperature_last[1, :], dest=rank - 1)  # Send top row

    # Update local grid points (Jacobi iteration)
    for i in range(1, local_rows + 1):
        for j in range(1, COLUMNS + 1):
            temperature[i, j] = 0.25 * (
                temperature_last[i + 1, j]
                + temperature_last[i - 1, j]
                + temperature_last[i, j + 1]
                + temperature_last[i, j - 1]
            )

    # Compute the local maximum difference (for convergence check)
    local_dt = np.max(
        np.abs(
            temperature[1 : local_rows + 1, 1 : COLUMNS + 1]
            - temperature_last[1 : local_rows + 1, 1 : COLUMNS + 1]
        )
    )

    # Update temperature_last
    temperature_last[1 : local_rows + 1, 1 : COLUMNS + 1] = temperature[
        1 : local_rows + 1, 1 : COLUMNS + 1
    ]

    # Global reduction to find the maximum difference across all processes
    dt = comm.allreduce(local_dt, op=MPI.MAX)

    if rank == 0 and iteration % 5 == 0:
        print(f"Iteration {iteration}, dt = {dt}")

    iteration += 1

# Gather the final temperature arrays to the root process for visualization
final_temperature = None
if rank == 0:
    final_temperature = np.zeros((ROWS + 2, COLUMNS + 2))

# Adjust the gathering process to account for the different row sizes
local_data = temperature[1 : local_rows + 1, :]  # Local portion excluding ghost cells

# Gather the data correctly at the root process
gathered_data = comm.gather(local_data, root=0)

if rank == 0:
    final_temperature[1 : local_rows + 1, :] = gathered_data[0]  # First process
    for i in range(1, size):
        start_row = i * (ROWS // size) + 1
        end_row = start_row + (ROWS // size)
        final_temperature[start_row:end_row, :] = gathered_data[i]

    # Visualize the result
    def plot_temperature(data):
        # plt.imshow(data, norm=cols.LogNorm(0.1, 50, clip=True))
        # plt.colorbar(label="Temperature")
        # plt.title("Temperature Distribution (Steady State)")
        # plt.xlabel("Columns")
        # plt.ylabel("Rows")
        # plt.savefig(f"Temperature_Plot_Parallel{iteration}.png")
        data.tofile("Parallel.out")

    plot_temperature(final_temperature)

if rank == 0:
    print(f"Converged after {iteration} iterations.")
