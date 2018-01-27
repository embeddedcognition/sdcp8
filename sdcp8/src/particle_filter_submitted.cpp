/*
###############################################
## AUTHOR: James Beasley                     ##
## DATE: July 8, 2017                        ##
## UDACITY SDC: Project 8 (Particle Filters) ##
###############################################
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

#include "particle_filter.h"

using namespace std;

//ingest a GPS position (represented by x and y), an initial heading (theta), and an array of uncertainties for these measurements
void ParticleFilter::init(double x, double y, double theta, double std[])
{
    //local vars
    default_random_engine gen;                                  //random number generator
    //standard deviations for x, y, and theta
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
    normal_distribution<double> dist_x(x, std_x);               //creates a normal (gaussian) distribution for x
    normal_distribution<double> dist_y(y, std_y);               //creates a normal (gaussian) distribution for y
    normal_distribution<double> dist_theta(theta, std_theta);   //creates a normal (gaussian) distribution for theta

    //set particle count
    num_particles = 10;

    //init particle and weight vectors based on the number of particles we wish to create
    particles.resize(num_particles);
    weights.resize(num_particles, 1.0); //initialize this to 1, otherwise we get garbage results and weights go to zero

    //init all particles to first position (based on estimates of x, y, theta and their uncertainties) and all weights to 1.
    for (int i = 0; i < particles.size(); i++)
    {
        //sample from the normal distributions above to initialize each attribute of the current particle
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
        particles[i].weight = 1.0;
    }

    //filter is now initialized
    is_initialized = true;
}

//update each particle's position estimate and account for uncertainty in the control inputs by introducing gaussian noise
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    //local vars
    default_random_engine gen;                          //random number generator
    double std_x = std_pos[0];                          //standard deviations for x, y, and theta
    double std_y = std_pos[1];
    double std_theta = std_pos[2];
    double predicted_x, predicted_y, predicted_theta;   //predicted positions

    //enumerate particles, updating each particle's position
    for (int i = 0; i < particles.size(); i++)
    {
        //deal with potential of yaw_rate being zero in computing state prediction for x, y, and theta
        if (fabs(yaw_rate) > 0.0001)
        {
            //compute predicted x, y, and theta via bicycle model when yaw rate is greater than zero
            predicted_x = particles[i].x + ((velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta)));
            predicted_y = particles[i].y + ((velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t))));
            predicted_theta = particles[i].theta + (yaw_rate * delta_t);
        }
        else
        {
            //compute predicted x, y, and theta via bicycle model when yaw rate is zero
            predicted_x = particles[i].x + (velocity * delta_t * cos(particles[i].theta));
            predicted_y = particles[i].y + (velocity * delta_t * sin(particles[i].theta));
            predicted_theta = particles[i].theta;
        }

        //to account for uncertainty in the control input, we need to add gaussian noise to the updated positions
        normal_distribution<double> dist_x(predicted_x, std_x);              //creates a normal (gaussian) distribution for x based on its updated position
        normal_distribution<double> dist_y(predicted_y, std_y);              //creates a normal (gaussian) distribution for y based on its updated position
        normal_distribution<double> dist_theta(predicted_theta, std_theta);  //creates a normal (gaussian) distribution for theta based on its updated position

        //assign new state
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

//unused
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    //https://discussions.udacity.com/t/c-help-with-dataassociation-method/291220
}

//update the weights for each particle
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks)
{
    //if we think of each particle as a potential heading/position of the car we're trying to localize, we want to transform each of the car's observations
    //into the perspective of that current particle (which is at a different heading/position), we can then understand how likely that particle is to being our true
    //position based on how close the observations that were transformed into the particle's perspective are to the landmarks on the map

    //local vars
    Map::single_landmark_s nearest_landmark;                    //the current nearest landmark to the current transformed observation
    double shortest_distance;                                   //the current shortest distance value
    double cur_distance;                                        //distance between the current landmark and the transformed observation
    double observation_weight;                                  //weight of the current observation
    double rotated_perspective_x, rotated_perspective_y;        //rotation values used during transformation step
    double translated_perspective_x, translated_perspective_y;  //rotation values used during transformation step
    double exp_value;                                           //exponent value for the constant "e"
    const double PI = 3.14159265358979;                         //PI

    //enumerate the particles
    for (int i = 0; i < particles.size(); i++)
    {
        //re-init particle weight to take new value
        particles[i].weight = 1.0;

        //enumerate the observations
        for (int j = 0; j < observations.size(); j++)
        {
            //reset for each observation (arbitrary value that we know is greater than a distance we'll see)
            shortest_distance = 1000;

            //transforms the current observation from the car's perspective to the particle's perspective
            //this will allow us to now have the observations in map coords and with respect to the particle's perspective
            //rotate the observation's (in vehicle coords) position with respect to the current particles heading (in map coords)
            rotated_perspective_x = (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
            rotated_perspective_y = (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);

            //translate the rotated observation based on the particle's current position
            translated_perspective_x = particles[i].x + rotated_perspective_x;
            translated_perspective_y = particles[i].y + rotated_perspective_y;

            //associate the transformed observation with its nearest landmark on the map (nearest neighbor)
            //eunumerate the landmarks
            for (int k = 0; k < map_landmarks.landmark_list.size(); k++)
            {
                //to speed nearest neighbor up, filter out landmarks that are more than sensor range away from the particle in the x and y direction
                //bounding box optimization: https://stackoverflow.com/questions/8690129/algorithm-to-find-all-points-on-a-2d-grid-some-distance-away-from-another-point
                if ((fabs(map_landmarks.landmark_list[k].x_f - particles[i].x) <= sensor_range) && (fabs(map_landmarks.landmark_list[k].y_f - particles[i].y) <= sensor_range))
                {
                    //compute distance between current landmark and transformed observation
                    cur_distance = dist(translated_perspective_x, translated_perspective_y, map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f);
                    //if the current distance is less than the shortest distance
                    if (cur_distance < shortest_distance)
                    {
                        //set the current landmark to the nearest landmark
                        nearest_landmark = map_landmarks.landmark_list[k];
                        //current distance is now the shortest distance
                        shortest_distance = cur_distance;
                    }
                }
            }
            //the current value is the nearest landmark

            //multiply in the weights of the transformed observations
            //compute exp value (exponent of constant "e")
            exp_value = (pow((translated_perspective_x - nearest_landmark.x_f), 2) / (2.0 * std_landmark[0] * std_landmark[0])) + (pow((translated_perspective_y - nearest_landmark.y_f), 2) / (2.0 * std_landmark[1] * std_landmark[1]));
            //multiply in the computed likelihood
            observation_weight = (1.0 / (2.0 * PI * std_landmark[0] * std_landmark[1])) * exp(-(exp_value));

            //multiply in the current observation's weight
            particles[i].weight *= observation_weight;
        }

        //we then establish a weight for the current particle that defines its likelihood of being the true heading/position of the car
        //the particle's final weight will be calculated as the product of each transformed observation's multivariate-gaussian probability
        //this weight tells us how important the particle it, the larger the weight, the more important it is
        //add this weight to the global weights vector for easy access when resampling
        weights[i] = particles[i].weight;
    }
}

//resample particles with replacement with probability proportional to their weight
void ParticleFilter::resample()
{
    //local vars
    random_device rd;
    mt19937 gen(rd());
    vector<Particle> resampled_particles;
    discrete_distribution<> resampling_distribution(weights.begin(), weights.end());

    //enumerate particles and resample with replacement
    for(int i = 0; i < particles.size(); i++)
    {
        //resample a particle
        Particle resampled_particle = particles[resampling_distribution(gen)];
        //add it to the vector of resampled particles
        resampled_particles.push_back(resampled_particle);
    }

    //overwrite current set of particles with the resampled particles
    particles = resampled_particles;
}

//unused
Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

//debug helper function
string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

//debug helper function
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

//debug helper function
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
