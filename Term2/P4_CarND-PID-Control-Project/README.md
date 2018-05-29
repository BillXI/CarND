# Controls PID

## By Xi Chen (May 29, 2018)

Self-Driving Car Engineer Nanodegree Program

Sorry for the delay, I was finishing my paper for the last two month, which is my degree requirement and then I was invited to present my poster for about two weeks extra.

Thanks for understanding!

---



## Basic Build Instructions

1. Clone this repo or unzip the downloaded file, go to the root directory.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./pid`. 

## Results:
I have tried the effect of the P, I, D component of the PID algorithm in their implementation. Here are the result.

* For only use the proportional control, it overshoots all the time, so the car swing around. [link](https://youtu.be/8nk2EFQ8AIY).
* For the integral control, I don't quite understand the purpose here, I now that it suppose to balance out the system bias, but the value is always go up, so it make the car just turn one direction constantly. I feel we should have used average CTE instead. [link](https://youtu.be/ieMTCqZYb_M).
* For the Differential control, it performs the best of the three, but it doesn't last long as soon as there's turn or curve. [link](https://youtu.be/aRTuXE967Lc)
* I use the this set of values to initiate the coefficient for the controller, [0.1, 0.0, 20.0], which performs the best. [link](https://youtu.be/SI5i84sWzdk)
 
 As showing in the last video, the simulator successfully and smoothly drove a whole lap. 

