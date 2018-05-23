#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;
    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // set state dimension and augmented dimension
    n_x_ = 5;
    n_aug_ = 7;
    lambda_ = 3 - n_aug_; // spreading parameter

    // initial state vector
    x_ = VectorXd(n_x_);

    // initial covariance matrix with identity matrix
    P_ = MatrixXd(n_x_, n_x_);
    P_.fill(0.0);
    P_.diagonal().fill(1.0);

    // initialize Unscented Transform weights of sigma points
    weights_ = VectorXd(2 * n_aug_ + 1);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    double weight = 0.5 / (n_aug_ + lambda_);
    for (int i = 1; i < 2 * n_aug_ + 1; i++) {
        weights_(i) = weight;
    }



    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.0; //30;
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1.0; //30;




    //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;
    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;
    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;
    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;
    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
    //***********************************//

    /**
    TODO:

    Complete the initialization. See ukf.h for other member properties.

    Hint: one or more values initialized above might be wildly off...
    */
    //set laser measurement dimension, laser can measure x and y, 2D
    const int n_z_laser = 2;

    //Laser measurement matrix
    H_ = MatrixXd(n_z_laser, n_x_);
    H_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;
    //measurement covariance matrix - laser
    R_laser_ = MatrixXd(n_z_laser, n_z_laser);
    R_laser_ << std_laspx_ * std_laspx_, 0,
            0, std_laspy_ * std_laspy_;


    //set radar measurement dimension, radar can measure r, phi, and r_dot, 3D
    const int n_z_radar = 3;
    // measurement noise covariance matrix
    R_radar_ = MatrixXd(n_z_radar, n_z_radar);
    R_radar_ << std_radr_ * std_radr_, 0, 0,
            0, std_radphi_ * std_radphi_, 0,
            0, 0, std_radrd_ * std_radrd_;

    is_initialized_ = false;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Make sure you switch between lidar and radar
    measurements.
    */
    if (!is_initialized_) {

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            // radar
            cout << "init radar ..." << endl;
            double rho = meas_package.raw_measurements_[0];
            double phi = meas_package.raw_measurements_[1];
            double rho_dot = meas_package.raw_measurements_[2];

            // convert radar from polar to cartesian coordinates
            // x_ is state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate]
            x_.fill(0.0);
            x_(0) = rho * cos(phi);
            x_(1) = rho * sin(phi);
            x_(3) = rho_dot * cos(phi);
            x_(4) = rho_dot * sin(phi);

        } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            // laser
            cout << "init laser ..." << endl;
            x_.fill(0.0);
            x_(0) = meas_package.raw_measurements_[0];
            x_(1) = meas_package.raw_measurements_[1];
        }

        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        return;
    }
    /*****************************************************************************
    *  Prediction
    ****************************************************************************/
    double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;    //dt: expressed in seconds
    time_us_ = meas_package.timestamp_;
    Prediction(delta_t);

    /*****************************************************************************
    *  Update
    ****************************************************************************/
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        UpdateRadar(meas_package);
    } else {
        // Lidar updates
        UpdateLidar(meas_package);
    }

    return;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    TODO:

    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */

    MatrixXd Xsig_aug;
    AugmentedSigmaPoints(&Xsig_aug);
    SigmaPointPrediction(Xsig_aug, &Xsig_pred_, delta_t);

    VectorXd x_pred;
    MatrixXd P_out;
    PredictMeanAndCovariance(&x_pred, &P_out);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */
    cout << "update laser ..." << endl;
    VectorXd z = meas_package.raw_measurements_;
    VectorXd z_pred = H_ * x_; // transformation
    VectorXd residual = z - z_pred; // Innovation = measurement residual

    // measurement covariance maxtrix
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_laser_;

    // Kalman gain calculation
    MatrixXd Si = S.inverse();
    MatrixXd K = P_ * Ht * Si;

    // state update
    x_ = x_ + (K * residual);

    // Covariance matrix update
    P_ = P_ - K * S * K.transpose();

    // Normalized Innovation Squared calculation
    NIS_laser_ = residual.transpose() * Si * residual;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    TODO:

    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */
    cout << "update radar ..." << endl;
    // dimension = 3: rho, phi, and rho_dot
    const int n_z = 3;
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);

    // measurement covariance maxtrix calculation
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);

    MatrixXd Zsig; // measurement model
    PredictRadarMeasurement(&z_pred, &S, Zsig);

    VectorXd x_pred;
    MatrixXd P_pred_out;

    VectorXd z = VectorXd(n_z);
    z = meas_package.raw_measurements_;
    UpdateState(Zsig, S, z_pred, z, &x_pred, &P_pred_out);

    NIS_radar_ = (z_pred - z).transpose() * S.inverse() * (z_pred - z);
}


/*******************************/
/**       Helper Functions
/** Borrowed from the exercise and modified
/*******************************/
void UKF::GenerateSigmaPoints(MatrixXd *Xsig_out) {

    //create sigma point matrix
    MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);

    //calculate square root of P
    MatrixXd A = P_.llt().matrixL();

    //set sigma points as columns of matrix Xsig
    Xsig.col(0) = x_;

    //calculate sigma points ...
    MatrixXd x_replicated = MatrixXd(n_x_, 2 * n_x_ + 1);
    x_replicated = x_.replicate(1, 5);

    Xsig.block(0, 1, n_x_, n_x_) = x_replicated + sqrt(lambda_ + n_x_) * A;
    Xsig.block(0, 6, n_x_, n_x_) = x_replicated - sqrt(lambda_ + n_x_) * A;

    //write result
    *Xsig_out = Xsig;

}

void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out) {
    //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.head(n_x_) = x_; // here is 5
    x_aug(5) = 0.0;
    x_aug(6) = 0.0; //set mean_a and mean_yawdd = 0

    // Create noise covariance matrix
    MatrixXd Q = MatrixXd(2, 2);
    Q << std_a_ * std_a_, 0,
            0, std_yawdd_ * std_yawdd_;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.block(0, 0, n_x_, n_x_) = P_;
    P_aug.block(n_x_, n_x_, 2, 2) = Q;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0) = x_aug;

    MatrixXd x_replicated = x_aug.replicate(1, n_aug_);  // reshape matrix

    Xsig_aug.block(0, 1, n_aug_, n_aug_) =
            x_replicated + sqrt(lambda_ + n_aug_) * L;

    Xsig_aug.block(0, n_aug_ + 1, n_aug_, n_aug_) =
            x_replicated - sqrt(lambda_ + n_aug_) * L;

    //write result
    *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd &Xsig_aug, MatrixXd *Xsig_out, double delta_t) {

    //create matrix with predicted sigma points as columns
    MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    //predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        //extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        } else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a * delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        //write predicted sigma point into right column
        Xsig_pred(0, i) = px_p;
        Xsig_pred(1, i) = py_p;
        Xsig_pred(2, i) = v_p;
        Xsig_pred(3, i) = yaw_p;
        Xsig_pred(4, i) = yawd_p;
    }

    //write result
    *Xsig_out = Xsig_pred;

}

void UKF::PredictMeanAndCovariance(VectorXd *x_pred_out, MatrixXd *P_out) {

    //predict state mean
    x_ = (Xsig_pred_ * weights_);

    //predicted state covariance matrix
    P_.fill(0.0);

    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        double angle_phi = x_diff(3);
        x_diff(3) = fmod(angle_phi, 2. * M_PI);

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }

    //write result
    *x_pred_out = x_;
    *P_out = P_;
}

void UKF::PredictRadarMeasurement(VectorXd *z_out, MatrixXd *S_out, MatrixXd &Zsig) {

    //set measurement dimension, radar can measure r, phi, and r_dot
    const int n_z = 3;

    Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
    Zsig.fill(0.0);

    //transform sigma points into measurement space
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;

        // measurement model
        if (fabs(p_x) < 1e-3)
            p_x = 1e-3;
        if (fabs(p_y) < 1e-3)
            p_y = 1e-3;
        Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                        //r
        Zsig(1, i) = atan2(p_y, p_x);                                 //phi
        Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }

//measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z, n_z);
    S.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        double angle_phi = z_diff(1);
        z_diff(1) = fmod(angle_phi, 2. * M_PI);

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    S = S + R_radar_;

    //write result
    *z_out = z_pred;
    *S_out = S;
}

void UKF::UpdateState(MatrixXd &Zsig, MatrixXd &S, VectorXd &z_pred, VectorXd &z, VectorXd *x_out, MatrixXd *P_out) {

    //set measurement dimension, radar can measure r, phi, and r_dot
    const int n_z = 3;

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        VectorXd z_diff = Zsig.col(i) - z_pred; //residual
        //angle normalization
        double angle_phi = z_diff(1);
        z_diff(1) = fmod(angle_phi, 2. * M_PI);

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        double angle_phi_diff = x_diff(3);
        x_diff(3) = fmod(angle_phi_diff, 2. * M_PI);

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    double angle_phi = z_diff(1);
    z_diff(1) = fmod(angle_phi, 2. * M_PI);

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    //write out result
    *P_out = P_;
    *x_out = x_;
}