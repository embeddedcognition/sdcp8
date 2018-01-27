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
void ParticleFilter::init(const double& x, const double& y, const double& theta, const vector<double>& sigma_pos)
{
    //if our vector of standard deviations is not the size expected, return and print an error
    if (sigma_pos.size() != 3)
    {
        cout << "Error: Expected 3 standard deviation values for init step, received" << sigma_pos.size() << "." << endl;
        return;
    }

    //local vars
    default_random_engine gen;                                  //random number generator
    //standard deviations for x, y, and theta
    double std_x = sigma_pos[0];
    double std_y = sigma_pos[1];
    double std_theta = sigma_pos[2];
    normal_distribution<double> dist_x(x, std_x);               //creates a normal (gaussian) distribution for x
    normal_distribution<double> dist_y(y, std_y);               //creates a normal (gaussian) distribution for y
    normal_distribution<double> dist_theta(theta, std_theta);   //creates a normal (gaussian) distribution for theta

    num_particles = 50;    //set particle count

    //init particle and weight vector based on the number of particles we wish to create
    particles.resize(num_particles);
    weights.resize(num_particles, 1.0); //initialize this to 1, otherwise we get garbage results and weights go to zero

    //init all particles to first position (based on estimates of x, y, theta and their uncertainties) and all weights to 1.
    for (Particle& cur_particle : particles)
    {
        //sample from the normal distributions above to initialize each attribute of the current particle
        cur_particle.x = dist_x(gen);
        cur_particle.y = dist_y(gen);
        cur_particle.theta = dist_theta(gen);
        cur_particle.weight = 1;
    }
}

//update each particle's position estimate and account for uncertainty in the control inputs by introducing gaussian noise
void ParticleFilter::prediction(const double& delta_t, const vector<double>& sigma_pos, const double& velocity, const double& yaw_rate)
{
    //if our vector of standard deviations is not the size expected, return and print an error
    if (sigma_pos.size() != 3)
    {
        cout << "Error: Expected 3 standard deviation values for prediction step, received" << sigma_pos.size() << "." << endl;
        return;
    }

    //local vars
    default_random_engine gen;                          //random number generator
    double std_x = sigma_pos[0];                        //standard deviations for x, y, and theta
    double std_y = sigma_pos[1];
    double std_theta = sigma_pos[2];
    double predicted_x, predicted_y, predicted_theta;   //predicted positions

    //enumerate particles, updating each particle's position
    for (Particle& cur_particle : particles)
    {
        //deal with potential of yaw_rate being zero in computing state prediction for x, y, and theta
        if (fabs(yaw_rate) > 0.0001)
        {
            //compute predicted x, y, and theta via bicycle model when yaw rate is greater than zero
            predicted_x = cur_particle.x + ((velocity / yaw_rate) * (sin(cur_particle.theta + (yaw_rate * delta_t)) - sin(cur_particle.theta)));
            predicted_y = cur_particle.y + ((velocity / yaw_rate) * (cos(cur_particle.theta) - cos(cur_particle.theta + (yaw_rate * delta_t))));
            predicted_theta = cur_particle.theta + (yaw_rate * delta_t);
        }
        else
        {
            //compute predicted x, y, and theta via bicycle model when yaw rate is zero
            predicted_x = cur_particle.x + (velocity * delta_t * cos(cur_particle.theta));
            predicted_y = cur_particle.y + (velocity * delta_t * sin(cur_particle.theta));
            predicted_theta = cur_particle.theta;
        }

        //to account for uncertainty in the control input, we need to add gaussian noise to the updated positions
        normal_distribution<double> dist_x(predicted_x, std_x);              //creates a normal (gaussian) distribution for x based on its updated position
        normal_distribution<double> dist_y(predicted_y, std_y);              //creates a normal (gaussian) distribution for y based on its updated position
        normal_distribution<double> dist_theta(predicted_theta, std_theta);  //creates a normal (gaussian) distribution for theta based on its updated position

        //assign new state
        cur_particle.x = (predicted_x + dist_x(gen));
        cur_particle.y = (predicted_y + dist_y(gen));
        cur_particle.theta = (predicted_theta + dist_theta(gen));
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
void ParticleFilter::updateWeights(const double& sensor_range, const std::vector<double>& sigma_landmark, const vector<LandmarkObs>& observations, const Map& map_landmarks)
{
    //if we think of each particle as a potential heading/position of the car we're trying to localize, we want to transform each of the car's observations
    //into the perspective of that current particle (which is at a different heading/position), we can then understand how likely that particle is to being our true
    //position based on how close the observations that were transformed into the particle's perspective are to the landmarks on the map

    //local vars
    LandmarkObs transformed_observation;        //an observation that has been transformed from the car's perspective to a particle's perspective
    Map::single_landmark_s nearest_landmark;    //nearest landmark to the current transformed observation
    double total_weight;                        //total weight of the current particle

    //clear global weights vector
    weights.clear();

    //enumerate the particles
    for (Particle& cur_particle : particles)
    {
        //init
        total_weight = 1;

        //enumerate the observations
        for (const LandmarkObs& cur_observation : observations)
        {
            //transform the current observation to the particle's perspective
            transform_observation_perspective(cur_particle, cur_observation, transformed_observation);
            //associate the transformed observation with its nearest landmark on the map (nearest neighbor)
            nearest_landmark = find_nearest_landmark(transformed_observation, map_landmarks);
            //multiply in the weights of the transformed observations
            total_weight *= compute_transformed_observation_weight(transformed_observation, nearest_landmark, sigma_landmark);
        }
        //we then establish a weight for the current particle that defines its likelihood of being the true heading/position of the car
        //the particle's final weight will be calculated as the product of each transformed observation's multivariate-gaussian probability
        //this weight tells us how important the particle it, the larger the weight, the more important it is
        //assign final weight
        cur_particle.weight = total_weight;
        //add this weight to the global weights vector for easy access when resampling
        weights.push_back(total_weight);
    }
    //https://discussions.udacity.com/t/update-step-confusion/247200/5
}

//transforms the current observation from the car's perspective to the particle's perspective
//this will allow us to now have the observations in map coords and with respect to the particle's perspective
void ParticleFilter::transform_observation_perspective(const Particle& particle, const LandmarkObs& observation, LandmarkObs& transformed_observation)
{
    //local vars
    double rotated_perspective_x, rotated_perspective_y;

    //rotate the observation's (in vehicle coords) position with respect to the current particles heading (in map coords)
    rotated_perspective_x = (cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);
    rotated_perspective_y = (sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);

    //translate the rotated observation based on the particle's current position
    transformed_observation.x = particle.x + rotated_perspective_x;
    transformed_observation.y = particle.y + rotated_perspective_y;
}

//find the landmark nearest to the transformed observation
Map::single_landmark_s ParticleFilter::find_nearest_landmark(const LandmarkObs& transformed_observation, const Map& map_landmarks)
{
    double smallest_distance = 100000000;       //set high to ensure we're able to loop
    double distance;                            //used in the loop
    Map::single_landmark_s nearest_landmark;    //nearest landmark

    //eunumerate the landmarks
    for (const Map::single_landmark_s& current_landmark : map_landmarks.landmark_list)
    {
        //compute distance between current landmark and transformed observation
        distance = dist(transformed_observation.x, transformed_observation.y, current_landmark.x_f, current_landmark.y_f);
        //if the distance is less than the smallest distance, set the current landmark to the nearest landmark
        if (distance < smallest_distance)
        {
            nearest_landmark = current_landmark;
        }
    }

    //the current value is the nearest landmark
    return nearest_landmark;
}

//compute the transformed observation's weight (likelihood)
double ParticleFilter::compute_transformed_observation_weight(const LandmarkObs& transformed_observation, const Map::single_landmark_s& nearest_landmark, const vector<double> sigma_landmark)
{
    //local vars
    double exp_value;

    //compute exp value
    exp_value = -((pow((transformed_observation.x - nearest_landmark.x_f), 2) / (2 * sigma_landmark[0] * sigma_landmark[0])) + (pow((transformed_observation.y - nearest_landmark.y_f), 2) / (2 * sigma_landmark[1] * sigma_landmark[1])));

    //return computed likelihood
    return (1 / (2 * PI * sigma_landmark[0] * sigma_landmark[1])) * exp(exp_value);
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
