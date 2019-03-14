import numpy as np
from scipy import linalg
from utils import *

def hat_operator(a):
	# return the skew symmetric matrix
	return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

def feature_to_world(features, wTc, M, b):
	# covert pixel to world coordinates
	result = np.ones((4,np.shape(features)[1]))
	result[0,:] = (features[0,:]-M[0,2])*b/(features[0,:]-features[2,:])
	result[1,:] = (features[1,:]-M[1,2])*(-M[2,3])/(M[1,1]*(features[0,:]-features[2,:]))
	result[2,:] = -M[2,3]/(features[0,:]-features[2,:])
	result = np.dot(wTc,result)
	return result

def get_jacobian(M, cTw, total_number_of_features, update_feature_index, prior):
	D = np.vstack((np.identity(3),np.zeros((1,3))))
	H = np.zeros((4*np.size(update_feature_index),3*total_number_of_features))
	for i in range(np.size(update_feature_index)):
		current_index = update_feature_index[i]
		H[i*4:(i+1)*4,current_index*3:(current_index+1)*3] =  np.dot(np.dot(np.dot(M,projection_derivative(np.dot(cTw,prior[:,i]))),cTw),D)
	return H

def projection_derivative(q):
	result = np.identity(4)
	result[2,2] = 0
	result[0,2] = -q[0]/q[2]
	result[1,2] = -q[1]/q[2]
	result[3,2] = -q[3]/q[2]
	result = result/q[2]
	return result

def projection(q):
	q = q/q[2,:]
	return q

if __name__ == '__main__':
	filename = "./data/0027.npz"
	t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)

	trajectory = np.zeros((4,4,np.size(t))) 	# imu pose over time
	#trajectory = np.zeros((4,4,1)) 	# imu pose over time
	#updated_trajectory = np.zeros((4,4,np.size(t))) 	# updated imu pose over time
	imu_mu_t_t = np.identity(4)					# mean
	imu_sigma_t_t = np.identity(6)				# covariance
	trajectory[:,:,0] = imu_mu_t_t				# initialize from origin
	#trajectory[:,:,0] = imu_mu_t_t	
	#updated_trajectory[:,:,0] = imu_mu_t_t		# initialize from origin

	M = np.block([[K[0:2,:], np.zeros((2,1))],[K[0:2,:], np.zeros((2,1))]])		# Stereo camera calibration matrix M
	M[2,3] = -K[0,0]*b
	V = 100
	landmark_mu_t = -1*np.ones((4,np.shape(features)[1]))	# mean	4*M
	landmark_sigma_t = np.identity(3*np.shape(features)[1])*V	# covariance	3M*3M
	D = np.vstack((np.identity(3),np.zeros((1,3))))
	D = np.kron(np.eye(np.shape(features)[1]),D)

	for i in range(1,np.size(t)):
		# (a) IMU Localization via EKF Prediction
		tau = t[0,i] - t[0,i-1]
		u_t = np.vstack((linear_velocity[:,i].reshape(3,1),rotational_velocity[:,i].reshape(3,1)))
		u_t_hat = np.block([[hat_operator(u_t[3:6,0]), u_t[0:3,0].reshape(3,1)],[np.zeros((1,3)), 0]])
		u_t_wedge = np.block([[hat_operator(u_t[3:6,0]), hat_operator(u_t[0:3,0])],[np.zeros((3,3)), hat_operator(u_t[3:6,0])]])
		imu_mu_t_t = np.dot(linalg.expm(-tau*u_t_hat), imu_mu_t_t)
		imu_sigma_t_t = np.dot(np.dot(linalg.expm(-tau*u_t_wedge),imu_sigma_t_t),np.transpose(linalg.expm(-tau*u_t_wedge))) + tau*tau*np.diag(np.random.normal(0,1,6))
		trajectory[:,:,i] = linalg.inv(imu_mu_t_t)
		#trajectory = np.dstack((trajectory, linalg.inv(imu_mu_t_t)))
		
		# (b) Landmark Mapping via EKF Update
		cTw = np.dot(cam_T_imu,imu_mu_t_t)	# world frame to camera frame
		wTc = linalg.inv(cTw)
		features_t = features[:,:,i]
		feature_index = np.array(np.where(np.sum(features_t[:,:],axis=0)!=-4))	# observed features
		update_feature_index = np.empty(0,dtype=int)
		update_feature = np.empty((4,0))

		if (np.size(feature_index)!=0):
			extracted_features_coor = features_t[:,feature_index].reshape(4,np.size(feature_index))
			extracted_features = feature_to_world(extracted_features_coor, wTc, M, b) # transform into world coordinates

			for j in range(np.size(feature_index)):
				current_index = feature_index[0,j]
				# if first seen, initialize landmark
				if (np.array_equal(landmark_mu_t[:,current_index],[-1,-1,-1,-1])):
					landmark_mu_t[:,current_index] = extracted_features[:,j]
				# else update landmark position
				else:
					update_feature_index = np.append(update_feature_index,current_index)
					update_feature = np.hstack((update_feature,extracted_features[:,j].reshape(4,1)))
			if (np.size(update_feature_index)!=0):
				mu_t_j = landmark_mu_t[:,update_feature_index].reshape((4,np.size(update_feature_index)))
				H = get_jacobian(M, cTw, np.shape(features)[1], update_feature_index, mu_t_j)
				K = np.dot(np.dot(landmark_sigma_t,np.transpose(H)),linalg.inv(np.dot(np.dot(H,landmark_sigma_t),np.transpose(H))+np.identity(4*np.size(update_feature_index))*V))
				z = features_t[:,update_feature_index].reshape((4,np.size(update_feature_index)))
				z_hat = np.dot(M,projection(np.dot(cTw,mu_t_j)))
				landmark_mu_t = (landmark_mu_t.reshape(-1,1,order='F') + np.dot(np.dot(D,K),(z-z_hat).reshape(-1,1,order='F'))).reshape(4,-1,order='F')
				landmark_sigma_t = np.dot((np.identity(3*np.shape(features)[1])-np.dot(K,H)),landmark_sigma_t)
			

		#visualize_trajectory_2d(trajectory,landmark_mu_t,show_ori=True)
	# (c) Visual-Inertial SLAM (Extra Credit)

	# You can use the function below to visualize the robot pose over time
	visualize_trajectory_2d(trajectory,landmark_mu_t,show_ori=True)
