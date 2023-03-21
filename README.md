# Eulerian Fluid Solver
This project extends Eulerian fluid solver in [taichi's stable fluid example](https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/stable_fluid.py) which uses semi-Lagrangian advection and Chorin-style projection with Jacobi iteration.

The extention includes MacCormack advection, conjugate gradients iteration with Jacobi preconditioner and multigrid preconditioner (MGPCG).

The implementation of MGPCG is based on [A Parallel Multigrid Poisson Solver for Fluids Simulation on Large Grids](https://www.math.ucla.edu/~jteran/papers/MST10.pdf).

## Usage
Create a Python virtual environment and install taichi
```angular2html
pip install taichi
```
Then run the code
```angular2html
python main.py
```
You can select iteration methods and turn on/off example mode by clicking the buttons. The number of iterations can be adjusted by pressing Up/Down on the keyboard.

When the example mode is off, you can create fluids by dragging in the window. (See [Example mode off](#example-mode-off))

Dye, velocity, curl, pressure visualization can be switched by pressing D, V, C, P on the keyboard.

## 2D Simulation
### Jacobi Iteration (20 times)
![Jacobi Iteration (20 times)](./results/results_jacobi_20_test_mode/video.gif)

### Jacobi Iteration (500 times)
![Jacobi Iteration (500 times)](./results/results_jacobi_500_test_mode/video.gif)

### Conjugate Gradients Iteration with Jacobi Preconditioner (20 times)
![Conjugate Gradients Iteration (20 times)](./results/results_cg_20_test_mode/video.gif)

### MGPCG (20 times)
![MGPCG (20 times)](./results/results_mgpcg_20_test_mode/video.gif)

### Example mode off
![Example mode off](./results/results_jacobi/video.gif)



