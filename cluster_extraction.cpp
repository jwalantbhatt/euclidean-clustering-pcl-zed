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
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>

//main
int 
main (int argc, char** argv)
{
  // Read in the cloud data
  pcl::PCDReader reader;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  reader.read ("table_scene_lms400.pcd", *cloud);
  std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.02f, 0.02f, 0.02f);
  vg.filter (*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*
  
  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PCDWriter writer;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);

  int i=0, nr_points = (int) cloud_filtered->points.size ();
  while (cloud_filtered->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    //std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;
  }
 

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (4000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);


  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_cluster(new pcl::visualization::PCLVisualizer("extracted clusters"));
  viewer_cluster->setBackgroundColor(0, 0, 0);

  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_bb(new pcl::visualization::PCLVisualizer("Bounding Boxes"));
  viewer_bb->setRepresentationToWireframeForAllActors();
 

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

	  viewer_cluster->addPointCloud<pcl::PointXYZ>(cloud_cluster, std::to_string(j));
	  viewer_cluster->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, std::to_string(j));
	  j++;
  }
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	viewer_cluster->addCoordinateSystem(1.0);
	viewer_cluster->initCameraParameters();

	viewer_bb->addCoordinateSystem(1.0);
	viewer_bb->initCameraParameters();
	viewer_bb->setRepresentationToWireframeForAllActors();

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	while (!viewer_cluster->wasStopped())
	{
		viewer_cluster->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
	while (!viewer_bb->wasStopped())
	{
		viewer_bb->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
  return (0);
}