#include <iostream>
#include <math.h>
#include <tuple>
#include <Eigen/Core> // Eigen Library
#include <Eigen/LU>   // Eigen Library

using namespace std;
using namespace Eigen;

float measurements[3] = { 1, 2, 3 };

void kalman_filter(MatrixXf& x, MatrixXf& P, MatrixXf& u, const MatrixXf& F, const MatrixXf& H, const MatrixXf& R, const MatrixXf& I)
{
	MatrixXf Z(1, 1);  
	MatrixXf y(1, 1);  
	MatrixXf S(1, 1);
	MatrixXf K(2, 1);
	for (int n = 0; n < sizeof(measurements) / sizeof(measurements[0]); n++) {

	//measurement stream
	Z << measurements[n];

	//calcualte measurement residual
	y << Z - (H * x);

	//map posterior state covariance to measurement space
	S << H * P * H.transpose() + R;

	//calculate Kalman Gain
	K << P * H.transpose() * S.inverse();

	//visualize iteration. Understand how each CoVariance changes shape
	cout << "Last guess : " << endl;	
	cout << "x: " << x << endl << endl;
	cout << "We were actually at..." << endl;
	cout << "Z: " << Z << endl << endl;
	cout << "Diff b/w last measurement and guessed state:" <<endl;
	cout << "y: " << y << endl << endl;
	cout << "Predicted state CoVariance (hopefully I tighten up a bit)  [variX correlation | correlation variY]" << endl;  
	cout << "P: " << P << endl << endl;
	cout << "Measurement CoVariance" << endl;
	cout << "S: " << S << endl;
	cout << "How much weight does each measurement hold?" << endl; 
	cout << "K: " << K << endl << endl;
	cout << "- - - - - " << endl;

        //update state
	x << x + (K * y);
	P << (I - (K * H)) * P;    

	//prediction
	x << (F * x) + u;
	P << F * P * F.transpose(); 
    }
}

int main()
{

    MatrixXf x(2, 1);// Initial state (location and velocity) 
    x << 0, //location
    	 0; //velocity
    MatrixXf P(2, 2);//Initial Uncertainty
    P << 100, 0, 
    	 0, 100; 
    MatrixXf u(2, 1);// External Motion
    u << 0,
    	 0; 
    MatrixXf F(2, 2);//Next State Function,  x' = x + x_dot*t, x_dot' = x_dot; 
    F << 1, 1,		// velocity equal to 1, and it remains constant
    	 0, 1; 
    MatrixXf H(1, 2);//Measurement Function
    H << 1,
    	 0; 
    MatrixXf R(1, 1); //Measurement Uncertainty
    R << 1;
    MatrixXf I(2, 2);// Identity Matrix
    I << 1, 0,
    	 0, 1; 

    kalman_filter(x, P, u, F, H, R, I);
    cout << "x= " << x << endl;
    cout << "P= " << P << endl;

    return 0;
}
