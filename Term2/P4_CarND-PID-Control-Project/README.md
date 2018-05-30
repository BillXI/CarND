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

## Hyperparameter (P,I,D) Tuning
I manually tuned the parameter. I first set all the hyperparameters to 0.0, and the results is terrible. Then I tried to set each hyperparameter individually to 1.0 and the rest two 0.0's to get a grasp of the effect of them on the simulation, and the results are showing in the youtube video below. 

I noticed that the differential coefficients playing a vital role in the smooth driving, so I start to increase the value over 1.0, then 5.0, then 10.0. On the other hand, I noticed the integral coefficient is just adding roughness to the driving, so I tried to decrease the value, eventually, I set to 0.0, the driving became much stable.

Even thought the differential coefficient did help the driving at the beginning, but it started behaving badly after the first turning. I knew it would slow to gain the stability, but the simulator need a "hard" control to keep the system back so it can have a small CTE against the reference track. The only coefficient can do that is proportional coefficient, so I increased it. It worked but still drove like a drunk driver. 

Therefore, I just slowly increase the differential coefficient and decrease the proportional coefficient while keep the integral coefficient zero. Eventually, I found the best results is with this set of values: [0.1, 0.0, 20.0].

## Results:
I have tried the effect of the P, I, D component of the PID algorithm in their implementation. Here are the result.

* For only use the proportional control, it overshoots all the time, so the car swing around. [link](https://youtu.be/8nk2EFQ8AIY).
* For the integral control, I don't quite understand the purpose here, I now that it suppose to balance out the system bias, but the value is always go up, so it make the car just turn one direction constantly. I feel we should have used average CTE instead. [link](https://youtu.be/ieMTCqZYb_M).
* For the Differential control, it performs the best of the three, but it doesn't last long as soon as there's turn or curve. [link](https://youtu.be/aRTuXE967Lc)
* I use the this set of values to initiate the coefficient for the controller, [0.1, 0.0, 20.0], which performs the best. [link](https://youtu.be/SI5i84sWzdk)
 
 As showing in the last video, the simulator successfully and smoothly drove a whole lap. 

