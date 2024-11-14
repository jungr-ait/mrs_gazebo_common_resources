/*
 * Software License Agreement (Modified BSD License)
 *
 *  Copyright (c) 2024
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PAL Robotics, S.L. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
#include <Eigen/Core>
#include <random>
#include <string>
#include <thread>

// GAZEBO
#include <boost/bind.hpp>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/physics/Collision.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/Noise.hh>

// ROS
#include <gazebo/rendering/DynamicLines.hh>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/UInt64.h>
#include <std_msgs/UInt64MultiArray.h>
#include <tf/transform_datatypes.h>
#include <uwb_msgs/TwoWayRangeStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <common.h>
#include <ignition/math.hh>

namespace gazebo {

class RosUwbTwr_plugin : public ModelPlugin
{
public:
  RosUwbTwr_plugin();
  virtual ~RosUwbTwr_plugin();

  struct config_t
  {
    float nlosSoftWallWidth = 0.1;
    float maxRange = 80.0;         // [m]
    float constantBias = 0.0;      // d: z = k*x + d  + noise [m]
    float distanceBasedBias = 1.0; // k: z = k*x + d  + noise [m]
    float twrRate = 10.0;          // [Hz]. Max possible number of Twr measurements per second
    float twrNoise = 0.1;          // standard deviation in [m]
    float refreshRateIDs = 0.1;    // [Hz] rate
    std::string twrTopic = "/twr";
    std::string deviceIdTopic = "/TWR_device_IDs";
    std::string requestIdTopic = "/TWR_request_IDs";
    std::string modelPrefix = "twr";
    std::string robot_namespace = "";
    uint deviceId = 0;
    bool allBeaconsAreLOS = false;
    bool useRangingIDs = true;
    bool useRoundRobin = true; // per update only one range to one device in the list is computed

    ignition::math::Vector3d antenna_offset = ignition::math::Vector3d::Zero;
  };

protected:
  virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
  void LoadThread();
  virtual void OnUpdate(const common::UpdateInfo &_info);

  ignition::math::Pose3d compute_antenna_pose(physics::EntityPtr entity_ptr,
                                              ignition::math::Vector3d const &offset);
  std::pair<double, bool> compute_range(ignition::math::Pose3d &pose_GA, size_t const other_id);
  void publish_range(double const range_raw,
                     double const range_gt,
                     bool const LOS,
                     size_t const other_id);
  double apply_twr_model(double const _gt_range);
  void get_sdf_params(sdf::ElementPtr _sdf);
  void callback_ranging_IDs(std_msgs::UInt64MultiArrayConstPtr msg);
  void callback_device_ID(std_msgs::UInt64ConstPtr msg);
  void callback_request_IDs(std_msgs::BoolConstPtr msg);
  void request_device_IDs();

  void add_device(const size_t device_id);
  void publish_id();
  void check_device_IDs();

private:
  config_t config_;
  std::string namespace_;
  physics::ModelPtr model_;
  physics::WorldPtr world_;
  ros::NodeHandlePtr node_handle_;
  std::thread deferred_load_thread_;
  ros::Publisher pub_twr_;
  uwb_msgs::TwoWayRangeStamped twr_msg_;

  event::ConnectionPtr update_connection_;

  common::Time last_pub_time_;
  std::default_random_engine random_generator_;
  std::normal_distribution<double> standard_normal_distribution_;
  size_t ranging_cnt_ = 0;
  physics::RayShapePtr firstRay,
      secondRay; /// rays between the antennas A and B, ray A->B, ray B->A

  /// Request and response device IDs;
  ros::Subscriber subRequestIDs_; /// subsrcibing to global topic /request_IDs - std_msgs::bool
  ros::Publisher pubRequestIDs_;  /// advertising at global topic /request_IDs - std_msgs::bool
  ros::Subscriber subDeviceID_;   /// subsrcibing to global topic /agent_ID - std_msgs::int
  ros::Publisher pubDeviceID_;    /// advertising at global topic /agent_ID - std_msgs::int
  common::Time last_ID_update_request_;

  struct deviceInfo_t
  {
    common::Time last_update;
    physics::EntityPtr entity_ptr;
    std::string model_name = "";
  };

  std::map<size_t, deviceInfo_t> dict_device_ids_;

  /// ranging ids
  std::vector<size_t> rangingIDs_;

}; // class UwbTwr_Plugin

RosUwbTwr_plugin::RosUwbTwr_plugin()
    : ModelPlugin()
{
  gzmsg << "RosUwbTwr_plugin::CTOR()" << std::endl;
}

RosUwbTwr_plugin::~RosUwbTwr_plugin()
{
  deferred_load_thread_.join();
  if (node_handle_) {
    node_handle_->shutdown();
  }
  update_connection_->~Connection();
}

void RosUwbTwr_plugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  model_ = _model;
  world_ = model_->GetWorld();
  last_pub_time_ = world_->SimTime();
  last_ID_update_request_ = last_pub_time_;
  gzmsg << "RosUwbTwr_plugin::Load(): ..." << std::endl;

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  random_generator_ = std::default_random_engine(seed);
  standard_normal_distribution_ = std::normal_distribution<double>(0.0, 1.0);
  // load parameters from Sdf
  get_sdf_params(_sdf);

  // Init ROS
  if (ros::isInitialized()) {
    ROS_INFO_STREAM("RosUwbTwr_plugin::Load(): ros initialized");
    // ros callback queue for processing subscription
    deferred_load_thread_ = std::thread(std::bind(&RosUwbTwr_plugin::LoadThread, this));
    deferred_load_thread_.detach();
    twr_msg_.header.frame_id = model_->GetName();
  } else {
    gzerr << "Not loading plugin since ROS hasn't been "
          << "properly initialized.  Try starting gazebo with ros plugin:\n"
          << "  gazebo -s libgazebo_ros_api_plugin.so\n";
  }
  gzwarn << "RosUwbTwr_plugin::Load(): URI=" << model_->URI().Str()
         << ", parent model name=" << model_->GetName() << ", this name=" << this->handleName
         << " DONE! \n";

  printf("[%s] RosUwbTwr_plugin: CNS plugin initialized\n", namespace_.c_str());

  // Listen to the update event. This event is broadcast every simulation iteration.
  update_connection_ = event::Events::ConnectWorldUpdateBegin(
      boost::bind(&RosUwbTwr_plugin::OnUpdate, this, _1));
}

void RosUwbTwr_plugin::LoadThread()
{
  size_t const buffer_size = 10;

  //ros::NodeHandle node_hdl_priv = ros::NodeHandle(config_.robot_namespace);

  ros::NodeHandle node_hdl("~");
  pub_twr_ = node_hdl.advertise<uwb_msgs::TwoWayRangeStamped>(config_.twrTopic, buffer_size);
  pubDeviceID_ = node_hdl.advertise<std_msgs::UInt64>("/twr_device_IDs", buffer_size);
  subDeviceID_ = node_hdl.subscribe("/twr_device_IDs",
                                    buffer_size,
                                    &RosUwbTwr_plugin::callback_device_ID,
                                    this);
  pubRequestIDs_ = node_hdl.advertise<std_msgs::Bool>("/twr_request_IDs", buffer_size);
  subRequestIDs_ = node_hdl.subscribe("/twr_request_IDs",
                                      buffer_size,
                                      &RosUwbTwr_plugin::callback_request_IDs,
                                      this);

  ROS_INFO_STREAM("RosUwbTwr_plugin::Load(): uwb twr plugin subscribed " << config_.twrTopic);

  request_device_IDs();
}

void RosUwbTwr_plugin::OnUpdate(const common::UpdateInfo &_info)
{
  float dt = (_info.simTime - last_pub_time_).Float();

  // time for an update?
  if (dt > (1.0f / config_.twrRate)) {
    last_pub_time_ = _info.simTime;
    ignition::math::Pose3d antenna_pose_1 = compute_antenna_pose(model_, config_.antenna_offset);

    // "with whom to range" IDs!
    std::vector<size_t> rangingIDs;
    if (config_.useRangingIDs && rangingIDs_.size()) {
      rangingIDs = rangingIDs_;
    } else {
      rangingIDs.reserve(dict_device_ids_.size());
      for (auto const &elem : dict_device_ids_) {
        rangingIDs.push_back(elem.first);
      }
    }

    // one by one (round robin) or all at once
    if (config_.useRoundRobin) {
      if (++ranging_cnt_ >= rangingIDs.size()) {
        ranging_cnt_ = 0;
      }
      // https://stackoverflow.com/a/21626211
      // Note that std::advance has void return type. If you want to return an advanced iterator, you can use std::next
      auto it = std::next(rangingIDs.begin(), ranging_cnt_);
      if (it != rangingIDs.end()) {
        std::pair<double, bool> meas = compute_range(antenna_pose_1, *it);
        if (meas.first > 0 && meas.first < config_.maxRange) {
          publish_range(apply_twr_model(meas.first), meas.first, meas.second, *it);
        }
      }
    } else {
      if (++ranging_cnt_ >= rangingIDs.size()) {
        // compute range to all known devices:
        ranging_cnt_ = 0;
        for (auto elem : rangingIDs) {
          std::pair<double, bool> meas = compute_range(antenna_pose_1, elem);
          if (meas.first > 0 && meas.first < config_.maxRange) {
            publish_range(apply_twr_model(meas.first), meas.first, meas.second, elem);
          }
        }
      }
    }
  }

  dt = (_info.simTime - last_ID_update_request_).Float();
  // time for an update?
  if (dt > (1.0f / config_.refreshRateIDs)) {
    request_device_IDs();
  }
}

ignition::math::Pose3d RosUwbTwr_plugin::compute_antenna_pose(physics::EntityPtr entity_ptr,
                                                              const ignition::math::Vector3d &offset)
{
  ignition::math::Pose3d antenna_pose;
  antenna_pose.Pos() = entity_ptr->WorldPose().Pos() + entity_ptr->WorldPose().Rot() * offset;
  antenna_pose.Rot() = entity_ptr->WorldPose().Rot();
  return antenna_pose;
}

std::pair<double, bool> RosUwbTwr_plugin::compute_range(ignition::math::Pose3d &pose_GA,
                                                        const size_t other_id)
{
  auto it = dict_device_ids_.find(other_id);
  if (it != dict_device_ids_.end()) {
    ignition::math::Pose3d pose_GB = compute_antenna_pose(it->second.entity_ptr,
                                                          config_.antenna_offset);

    double distance = pose_GA.Pos().Distance(pose_GB.Pos());
    bool LOS = true;
    if (!config_.allBeaconsAreLOS) {
      //We check if a ray can reach the anchor:
      double distanceToObstacleFromA;
      std::string obstacleName;

      ignition::math::Vector3d p_AB_in_G_normed = (pose_GB.Pos() - pose_GA.Pos()).Normalize();
      firstRay->Reset();
      firstRay->SetPoints(pose_GA.Pos(), pose_GB.Pos());
      firstRay->GetIntersection(distanceToObstacleFromA, obstacleName);
      if (obstacleName.compare("") != 0) {
        // There is an obstacle:
        // We use a second ray to measure the distance from anchor to tag ot compute the width of the obstacle

        double distanceToObstacleFromB;
        std::string otherObstacleName;

        this->secondRay->Reset();
        this->secondRay->SetPoints(pose_GB.Pos(), pose_GA.Pos());
        this->secondRay->GetIntersection(distanceToObstacleFromB, otherObstacleName);
        double obstacle_width = distance - (distanceToObstacleFromA + distanceToObstacleFromB);
        if (obstacle_width > config_.nlosSoftWallWidth
            || obstacleName.compare(otherObstacleName) != 0) {
          //We try to find a rebound to reach the B from A
          LOS = false;
        }
      }
    }
    return std::make_pair(distance, LOS);
  }
  return std::make_pair(0, false);
}

void RosUwbTwr_plugin::publish_range(const double range_raw,
                                     const double range_gt,
                                     const bool LOS,
                                     const size_t other_id)
{
  twr_msg_.range_corr = range_gt;
  twr_msg_.range_raw = range_raw;
  twr_msg_.LOS = LOS;
  twr_msg_.R = config_.twrNoise * config_.twrNoise;
  twr_msg_.UWB_ID1 = config_.deviceId;
  twr_msg_.UWB_ID2 = other_id;
}

double RosUwbTwr_plugin::apply_twr_model(const double _gt_range)
{
  double noise = config_.twrNoise * standard_normal_distribution_(random_generator_);
  return _gt_range * config_.distanceBasedBias + config_.constantBias + noise;
}

void RosUwbTwr_plugin::get_sdf_params(sdf::ElementPtr _sdf)
{
  getSdfParam(_sdf, "nlosSoftWallWidth", config_.nlosSoftWallWidth, 0.1f);
  getSdfParam(_sdf, "maxRange", config_.maxRange, 80.0f);
  getSdfParam(_sdf, "constantBias", config_.constantBias, 0.0f);
  getSdfParam(_sdf, "distanceBasedBias", config_.distanceBasedBias, 1.0f);
  getSdfParam(_sdf, "twrRate", config_.twrRate, 50.0f);
  getSdfParam(_sdf, "twrNoise", config_.twrNoise, 0.1f);
  getSdfParam(_sdf, "refreshRateIDs", config_.refreshRateIDs, 0.1f);
  getSdfParam(_sdf, "twrTopic", config_.twrTopic, std::string("/twr"));
  getSdfParam(_sdf, "deviceIdTopic", config_.deviceIdTopic, std::string("/TWR_device_IDs"));
  getSdfParam(_sdf, "requestIdTopic", config_.requestIdTopic, std::string("/TWR_request_IDs"));
  getSdfParam(_sdf, "modelPrefix", config_.modelPrefix, std::string("twr"));
  getSdfParam(_sdf, "robot_namespace", config_.robot_namespace, std::string(""));
  getSdfParam(_sdf, "deviceId", config_.deviceId, uint(0));
  getSdfParam(_sdf, "allBeaconsAreLOS", config_.allBeaconsAreLOS, false);
  getSdfParam(_sdf, "useRangingIDs", config_.useRangingIDs, true);
  getSdfParam(_sdf, "useRoundRobin", config_.useRoundRobin, true);
  {
    float x, y, z;
    getSdfParam(_sdf, "antenna_offset_x", x, 0.0f);
    getSdfParam(_sdf, "antenna_offset_x", y, 0.0f);
    getSdfParam(_sdf, "antenna_offset_x", z, 0.0f);
    config_.antenna_offset.Set(x, y, z);
  }
}

void RosUwbTwr_plugin::callback_ranging_IDs(std_msgs::UInt64MultiArrayConstPtr msg)
{
  rangingIDs_.clear();
  if(msg->data.size()) {
    rangingIDs_.reserve(msg->data.size());
    for(auto id : msg->data) {
      rangingIDs_.push_back((uint)id);
    }
  }
}

void RosUwbTwr_plugin::callback_device_ID(std_msgs::UInt64ConstPtr msg)
{
  if (msg->data != config_.deviceId) {
    if (dict_device_ids_.find(msg->data) == dict_device_ids_.end()) {
      // new device found...
      add_device(msg->data);
    } else {
      // update timestamp of known device...
      dict_device_ids_.find(msg->data)->second.last_update = world_->SimTime();
    }
  }
}

void RosUwbTwr_plugin::callback_request_IDs(std_msgs::BoolConstPtr msg)
{
  if (msg->data) {
    last_ID_update_request_ = world_->SimTime();
    publish_id();
  }
}

void RosUwbTwr_plugin::request_device_IDs()
{
  std_msgs::Bool request;
  request.data = true;
  pubRequestIDs_.publish(request);
}

void RosUwbTwr_plugin::add_device(const size_t device_id)
{
  deviceInfo_t info;
  info.last_update = world_->SimTime();
  info.model_name = "";

  // find the Gazebo model name:
  std::string model_name_prefix = std::string(config_.modelPrefix + "_" + std::to_string(device_id));
  for (auto &model_ptr : world_->Models()) {
    if (model_ptr->GetName().find(model_name_prefix) != std::string::npos) {
      info.model_name = model_ptr->GetName();
      info.entity_ptr = model_ptr;
      break;
    } else if (model_ptr != this->model_) {
      for (auto &link_ptr : model_ptr->GetLinks()) {
        //ROS_DEBUG("UwbTwr_Plugin[%u]::add_device([%zu]): link name=%s",
        //         config_.deviceId,
        //         device_id,
        //         link_ptr->GetName().c_str());
        if (link_ptr->GetName().find(model_name_prefix) != std::string::npos) {
          info.model_name = link_ptr->GetName();
          info.entity_ptr = link_ptr;
          break;
        }
      }
    }
  }
  if (!info.model_name.empty()) {
    dict_device_ids_.insert({device_id, info});
    ROS_INFO("UwbTwr_Plugin[%u]::add_device([%zu]): new device [%s] added",
             config_.deviceId,
             device_id,
             info.model_name.c_str());
  } else {
    //ROS_DEBUG("UwbTwr_Plugin[%u]::add_device([%zu]): model_name empty for prefix [%s]!",
    //          config_.deviceId,
    //          device_id,
    //          model_name_prefix.c_str());
  }
}

void RosUwbTwr_plugin::publish_id()
{
  std_msgs::UInt64 response;
  response.data = config_.deviceId;
  pubDeviceID_.publish(response);
}

void RosUwbTwr_plugin::check_device_IDs()
{
  std::vector<size_t> removeIDs;
  auto curr_time = world_->SimTime();
  for(auto elem : dict_device_ids_) {
    auto dt = curr_time - elem.second.last_update;
    if (dt.Float() > 0.5 / config_.refreshRateIDs) {
      removeIDs.push_back(elem.first);
    }
  }
  for(auto id : removeIDs) {
    auto it = dict_device_ids_.find(id);
    if (it != dict_device_ids_.end()) {
      dict_device_ids_.erase(it);
      ROS_INFO("UwbTwr_Plugin[%u]::check_device_IDs: device with ID [%zu] removed by ",
               config_.deviceId,
               id);
    }
  }
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(RosUwbTwr_plugin)
} // namespace gazebo
