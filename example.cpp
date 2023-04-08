#include <iostream>
#include <vector>
#include <random>
#include <array>
#include <list>
#include <unordered_set>

#include "nanoflann.hpp"
#include "spacecol.hpp"

int main() {
    const int K = 3; //dimension
    using real_t = double; //floating point datatype
    PointGraph<real_t, K> tree_nodes; //data structure for nodes
    tree_nodes.add_node(-1, point_t<real_t, K>({0.5, 0.5, 0.5}));  //inserting root node
    PointCloud<real_t, K> attraction_points; //data structure for attraction points
    attraction_points.points.insert(attraction_points.points.end(), 13000, point_t<real_t, K>({0, 0, 0})); //allocating space for attraction points
    generateRandomPointCloudRange<real_t, K, PointCloud<real_t, K>>(attraction_points, 0, 13000, 2); // random points generated in [0, 1]^K 
    //space colonization
    colonize<real_t, K>(tree_nodes, attraction_points, 1e-3, 1e-3, 0.4, 4, "snapshots/snap", 3200, 3); //make sure that the snapshot folder already exists
    tree_nodes.to_file("snapshots/snap-final.txt"); //save final state to file
}
