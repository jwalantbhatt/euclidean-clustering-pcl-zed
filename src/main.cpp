
// All includes
#include <sl/Camera.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iostream>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <boost/thread/thread.hpp>
#include <thread>
#include <mutex>

// Undef on Win32 min/max for PCL
#ifdef _WIN32
#undef max
#undef min
#endif

// Namespace
using namespace sl;
using namespace std;

// Global instance (ZED, Mat, callback)
Camera zed;
Mat data_cloud;
std::thread zed_callback;
std::mutex mutex_input;
bool stop_signal;
bool has_data;

// Sample functions
void startZED();
void run();
void closeZED();

// Main process
int main(int argc, char **argv) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->points.resize(zed.getResolution().area());


	// Declare and Set Viewers

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_cluster(new pcl::visualization::PCLVisualizer("extracted clusters"));
	viewer_cluster->setBackgroundColor(0, 0, 0);

	viewer_cluster->addCoordinateSystem(1.0);
	viewer_cluster->initCameraParameters();
	viewer_cluster->setRepresentationToWireframeForAllActors();

	// Set configuration parameters
	InitParameters init_params;
	init_params.camera_resolution = RESOLUTION_VGA;
	if (argc == 2)
		init_params.svo_input_filename = argv[1];
	init_params.coordinate_units = UNIT_METER;
	init_params.coordinate_system = COORDINATE_SYSTEM_RIGHT_HANDED_Y_UP;
	init_params.depth_mode = DEPTH_MODE_PERFORMANCE;

	// Open the camera
	ERROR_CODE err = zed.open(init_params);
	if (err != SUCCESS) {
		cout << errorCode2str(err) << endl;
		zed.close();
		return 1;
	}

	// Allocate PCL point cloud at the resolution
	pcl::PointCloud<pcl::PointXYZ>::Ptr p_pcl_point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	p_pcl_point_cloud->points.resize(zed.getResolution().area());

	// Start ZED callback
	startZED();

	// Loop infinitely
	while (1) {

		// Try to lock the data if possible (not in use). Otherwise, do nothing.
		if (mutex_input.try_lock()) {
			float *p_data_cloud = data_cloud.getPtr<float>();
			int index = 0;

			// Check and adjust points for PCL format
			for (auto &it : p_pcl_point_cloud->points) {
				float X = p_data_cloud[index];
				if (!isValidMeasure(X)) // Checking if it's a valid point
					it.x = it.y = it.z = 0; 
				else {
					it.x = X;
					it.y = p_data_cloud[index + 1];
					it.z = p_data_cloud[index + 2];
				}
				index += 4;
			}

			// Unlock data and update Point cloud
			mutex_input.unlock();

			// Create the filtering object: downsample the dataset using a leaf size of 1cm
			pcl::VoxelGrid<pcl::PointXYZ> vg;
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
			vg.setInputCloud(p_pcl_point_cloud);
			vg.setLeafSize(0.10f, 0.10f, 0.10f); // leaf sife
			vg.filter(*cloud_filtered);
			//std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl; 

			// Create the segmentation object for the planar model and set all the parameters
			pcl::SACSegmentation<pcl::PointXYZ> seg;
			pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
			pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());
			pcl::PCDWriter writer;
			seg.setOptimizeCoefficients(true);
			seg.setModelType(pcl::SACMODEL_PLANE);
			seg.setMethodType(pcl::SAC_RANSAC);
			seg.setMaxIterations(100);
			seg.setDistanceThreshold(0.02);
			
			int i = 0, nr_points = (int)cloud_filtered->points.size();
			while (cloud_filtered->points.size() > 0.3 * nr_points)
			{
				// Segment the largest planar component from the remaining cloud
				seg.setInputCloud(cloud_filtered);
				seg.segment(*inliers, *coefficients);
				if (inliers->indices.size() == 0)
				{
					std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
					break;
				}

				// Extract the planar inliers from the input cloud
				pcl::ExtractIndices<pcl::PointXYZ> extract;
				extract.setInputCloud(cloud_filtered);
				extract.setIndices(inliers);
				extract.setNegative(false);

				// Get the points associated with the planar surface
				extract.filter(*cloud_plane);
				//std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << std::endl;

				// Remove the planar inliers, extract the rest
				extract.setNegative(true);
				extract.filter(*cloud_f);
				*cloud_filtered = *cloud_f;
			}

			// Creating the KdTree object for the search method of the extraction
			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
			tree->setInputCloud(cloud_filtered);

			std::vector<pcl::PointIndices> cluster_indices;
			pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
			ec.setClusterTolerance(0.20); // 20cm
			ec.setMinClusterSize(70);     
			ec.setMaxClusterSize(210);
			ec.setSearchMethod(tree);
			ec.setInputCloud(cloud_filtered);
			ec.extract(cluster_indices);

			//clear clouds
			viewer_cluster->removeAllShapes();
			viewer_cluster->removeAllPointClouds();
			viewer_cluster->setRepresentationToWireframeForAllActors();

			int j = 0;
			for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
				for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
					cloud_cluster->points.push_back(cloud_filtered->points[*pit]); //*
				cloud_cluster->width = cloud_cluster->points.size();
				cloud_cluster->height = 1;
				cloud_cluster->is_dense = true;
				pcl::PointXYZ minPt, maxPt;

				pcl::getMinMax3D(*cloud_cluster, minPt, maxPt);
				std::cout << "Max x: " << maxPt.x << std::endl;
				std::cout << "Max y: " << maxPt.y << std::endl;
				std::cout << "Max z: " << maxPt.z << std::endl;
				std::cout << "Min x: " << minPt.x << std::endl;
				std::cout << "Min y: " << minPt.y << std::endl;
				std::cout << "Min z: " << minPt.z << std::endl;
				float x_min = minPt.x;
				float x_max = maxPt.x;
				float y_min = minPt.y;
				float y_max = maxPt.y;
				float z_min = minPt.z;
				float z_max = maxPt.z;
				viewer_cluster->setRepresentationToWireframeForAllActors();
				viewer_cluster->addCube(x_min, x_max, y_min, y_max, z_min, z_max, 1.0, 0, 0, std::to_string(j+50), 0);
				viewer_cluster->setRepresentationToWireframeForAllActors();
				viewer_cluster->addPointCloud<pcl::PointXYZ>(cloud_cluster, std::to_string(j));
				viewer_cluster->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, std::to_string(j));
				viewer_cluster->setRepresentationToWireframeForAllActors();
				//std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size() << " data points." << std::endl;
				j++;
			}
			
			viewer->updatePointCloud(p_pcl_point_cloud, "cloud");
			viewer->spinOnce(5000);
			viewer_cluster->spinOnce(5000);
			viewer_cluster->setRepresentationToWireframeForAllActors();
			//Sleep(5000);
		}
		else
			sleep_ms(1);
	}

	// Close the viewer
	viewer->close();

	// Close the zed
	closeZED();

	return 0;
}

//This functions start the ZED's thread that grab images and data.
void startZED() {
	// Start the thread for grabbing ZED data
	stop_signal = false;
	has_data = false;
	zed_callback = std::thread(run);

	//Wait for data to be grabbed
	while (!has_data)
		sleep_ms(1);
}

//This function loops to get the point cloud from the ZED. It can be considered as a callback.
void run() {
	while (!stop_signal)
	{
		if (zed.grab(SENSING_MODE_STANDARD) == SUCCESS)
		{
			mutex_input.lock(); // To prevent from data corruption
			zed.retrieveMeasure(data_cloud, MEASURE_XYZRGBA);
			mutex_input.unlock();
			has_data = true;
		}
		sleep_ms(1);
	}
}

//This function frees and close the ZED, its callback(thread) and the viewer
void closeZED() {
	// Stop the thread
	stop_signal = true;
	zed_callback.join();
	zed.close();
}




