/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
//#include <mpif-sizeof.h>

#include "particle_filter.h"

#define EPS 0.00001
// for value too small

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    if(is_initialized) return;

    num_particles = 100; // 20;

    // creating normal distribution for the noise
    default_random_engine gen;
    normal_distribution<double> ndist_x(x, std[0]), ndist_y(x, std[1]), ndist_theta(theta, std[2]);

    // generating particles
    for(int i=0; i<num_particles; i++){
        Particle particle;
        particle.id = i;
        particle.x = ndist_x(gen);
        particle.y = ndist_y(gen);
        particle.theta = ndist_theta(gen);
        particle.weight = 1.0;

        particles.push_back(particle);
    }

    cout << "Particles are initiated" << endl;

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    // Creating Gaussian noise for measurement
    default_random_engine gen;
    normal_distribution<double> ndist_x(0.0, std_pos[0]), ndist_y(0.0, std_pos[1]), ndist_theta(0.0, std_pos[1]);

    /* Estimate the new state
     * This part from the "motion models" notes.
     * update the position x,y
     */
    for(int i=0; i<num_particles; i++){
        double theta = particles[i].theta;
        if(fabs(yaw_rate)<EPS) { // when yar is not changing
            particles[i].x += velocity * delta_t * cos( theta ) + ndist_x(gen);
            particles[i].y += velocity * delta_t * sin( theta ) + ndist_y(gen);
            //particles[i].theta += ndist_theta(gen);
        }

        else{
            particles[i].x += velocity / yaw_rate * ( sin( theta + yaw_rate * delta_t ) - sin( theta ) ) + ndist_x(gen);
            particles[i].y += velocity / yaw_rate * ( cos( theta ) - cos( theta + yaw_rate * delta_t ) ) + ndist_y(gen);
            particles[i].theta += yaw_rate * delta_t + ndist_theta(gen);
        }
    }



}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

    for (int i = 0; i < observations.size(); i++) {
        int map_id;
        // set the best distance as big as possible, will be replace by the better candidate.
        double min_distance = numeric_limits<double>::max();

        // calculate the Euclidean distance
        for (int j = 0; j < predicted.size(); j++) {
            double distance;
            distance = sqrt(
                    pow((observations[i].x - predicted[j].x), 2) +
                    pow((observations[i].y - predicted[j].y), 2));


            if(distance < min_distance){
                min_distance = distance;
                map_id = predicted[j].id;
            }
        }

        observations[i].id = map_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // collect all the statistics
    double var_x = pow(std_landmark[0], 2);
    double var_y = pow(std_landmark[1], 2);
    double cov_xy = std_landmark[0] * std_landmark[1];

    // Transform observation coordinates.


    for (int i = 0; i < num_particles; i++) {
        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;

        // find landmarks in particle's range;
        vector<LandmarkObs> landmarksInRange;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            double landmark_x = map_landmarks.landmark_list[j].x_f;
            double landmark_y = map_landmarks.landmark_list[j].y_f;
            int landmark_id = map_landmarks.landmark_list[j].id_i;
            double landmark_distance = sqrt(pow((x-landmark_x),2) + pow((y-landmark_y),2));
            // if in the sensor range append to the landmarkInRange list
            if(landmark_distance <= sensor_range) {
                landmarksInRange.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
            }

        }

        // transform vehicle's observation to global coordinates
        vector<LandmarkObs> transformedObservations;
        for(unsigned int j = 0; j < observations.size(); j++) {
            double transformed_x = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
            double transformed_y = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
            transformedObservations.push_back(LandmarkObs{ observations[j].id, transformed_x, transformed_y });
        }

        // observation association to landmark.
        dataAssociation(landmarksInRange, transformedObservations);

        // reset weight.
        particles[i].weight = 1.0;

        // calculate weights
        for (int j = 0; j < transformedObservations.size(); j++) {
            double observation_x = transformedObservations[j].x;
            double observation_y = transformedObservations[j].y;
            int landmark_id = transformedObservations[j].id;
            double landmark_x, landmark_y;
            int k=0;
            bool is_found = false;
            while(!is_found && k < landmarksInRange.size()){
                if(landmarksInRange[k].id == landmark_id){
                    is_found = true;
                    landmark_x = landmarksInRange[k].x;
                    landmark_y = landmarksInRange[k].y;
                }
                k++;
            }
            // calculate weight
            double d_x_squared = pow((observation_x - landmark_x),2);
            double d_y_squared = pow((observation_y - landmark_y),2);
            double weight;
            weight = ( 1/(2*M_PI*cov_xy)) * exp( -( d_x_squared/(2*var_x) + (d_y_squared/(2*var_y)) ) );
            if(weight==0){
                particles[i].weight *= EPS;
            } else{
                particles[i].weight *= weight;
            }

        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // find the max weight.
    vector<double> weights;
    double weight_max = numeric_limits<double>::min();
    for (int i = 0; i < num_particles; i++) {
        weights.push_back(particles[i].weight);
        if(particles[i].weight > weight_max){
            weight_max = particles[i].weight;
        }
    }

    // create distribution
    uniform_real_distribution<double> dist_doouble(0, weight_max);
    uniform_int_distribution<int> dist_int(0, num_particles-1);
    default_random_engine gen;

    int index = dist_int(gen);
    double beta = 0.0;

    // resample process
    vector<Particle> temp_particles;
    for (int i = 0; i < num_particles; i++) {
        beta += dist_doouble(gen) * 2.0;
        while(beta > weights[index]){
            beta -= weights[index];
            index = (index + 1)% num_particles;
        }
        temp_particles.push_back(particles[index]);
    }
    particles = temp_particles;


//    default_random_engine gen;
//
//    // initiate a PMF for weights so generate particles.
//    discrete_distribution<> weights_pmf(weights.begin(), weights.end());
//
//    // initiate a temperary array for store the new particle.
//    vector<Particle> temp_particles;
//
//    // resample process.
//    for(int i = 0; i < num_particles; i++) {
//        temp_particles.push_back(particles[weights_pmf(gen)]);
//    }
//
//    // assigne the temperary array of particles to the particle.
//    particles = temp_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
