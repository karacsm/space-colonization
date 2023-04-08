#include <vector>
#include <random>
#include <array>
#include <list>
#include <fstream>
#include <string>

template<typename T, int K>
using point_t = std::array<T, K>;

template<typename T, int K>
class PointCloud {
    public:
    
    std::vector<point_t<T, K>> points;

    inline size_t kdtree_get_point_count() const { return points.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return points[idx][dim];
    }

    inline point_t<T, K> kdtree_get_pt(const size_t idx) const
    {
        return points[idx];
    }

    inline point_t<T, K>& coords_at(const size_t idx) {
        return points[idx];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

template<typename T, int K>
class PointGraph {
    public:
    struct PointGraphDataNode {
        int root_node {-1};
        std::list<int> children;
        point_t<T, K> coords; 
    };
    std::vector<PointGraphDataNode> data_nodes;

    inline size_t kdtree_get_point_count() const { return data_nodes.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return data_nodes[idx].coords[dim];
    }

    inline point_t<T, K> kdtree_get_pt(const size_t idx) const
    {
        return data_nodes[idx].coords;
    }

    inline point_t<T, K>& coords_at(const size_t idx) {
        return data_nodes[idx].coords;
    }

    inline int add_node(int root, const point_t<T, K> &coords) {
        int new_id = data_nodes.size();
        PointGraphDataNode new_node;
        new_node.root_node = root;
        new_node.coords = coords;
        data_nodes.push_back(new_node);
        if (root >= 0) data_nodes[root].children.push_back(new_id);
        return new_id;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }

    void to_file(const std::string& filename) {
        std::ofstream file;
        file.open(filename);
        file << data_nodes.size() << "\n";
        file << K << "\n";
        for (const PointGraphDataNode& node : data_nodes) {//K coords, root, number of children, children
            for (int i = 0; i < K; i++) {
                file << std::scientific << node.coords[i] << "\t";
            }
            file << node.root_node << "\t";
            file << node.children.size() << "\t";
            for (const int& i : node.children) {
                file << i << "\t";
            }
            file << "\n";
        }
        file.close();
    }
};

template<typename T, int K, typename PointDataStructure>
void generateRandomPointCloudRange(PointDataStructure &cloud, int start, int N, int seed = 0) {
    std::mt19937 rand(seed);
    std::uniform_real_distribution<T> dist(0, 1);
    for (int i = start; i < std::min<int>(start + N, cloud.kdtree_get_point_count()); i++) {
        point_t<T, K> point;
        for (int j = 0; j < K; j++) {
            cloud.coords_at(i)[j] = dist(rand);
        }
    }
}

template<typename T, int K, typename KDTree_index>
std::pair<size_t, T> get_closest_point(const KDTree_index &index, const point_t<T, K> &x) {
    const size_t               num_results = 1;
    size_t                     ret_index;
    T                          out_dist_sqr;
    nanoflann::KNNResultSet<T> resultSet(num_results);
    resultSet.init(&ret_index, &out_dist_sqr);
    T querry_point[K];
    for (int i = 0; i < K; i++) querry_point[i] = x[i];
    index.findNeighbors(resultSet, querry_point, {10});
    return {ret_index, out_dist_sqr};
}

template<typename T, int K>
std::array<T, K> normalize_vector(const std::array<T, K> &vector) {
    std::array<T, K> normed_vec; for (int i = 0; i < K; i++) normed_vec[i] = std::fabs(vector[i]);
    T m = *std::min_element(normed_vec.begin(), normed_vec.end());
    if (m == 0) {
        return normed_vec;
    }
    for (int i = 0; i < K; i++) normed_vec[i] = vector[i] / m;
    T r = 0;
    for (int i = 0; i < K; i++) r += normed_vec[i] * normed_vec[i];
    r = std::sqrt(r);
    for (int i = 0; i < K; i++) normed_vec[i] /= r;
    return normed_vec;
}

template<typename T, int K>
std::array<T, K> axpy(T a, const std::array<T, K> &x, const std::array<T, K> &y) {
    std::array<T, K> out_vec;
    for (int i = 0; i < K; i++) out_vec[i] = a * x[i] + y[i];
    return out_vec;
}

//helper functions for void colonize(...)
template<typename real_t, int K, typename PointDataStructure, typename kd_graph_tree_t>
std::vector<std::list<int>> prune_attraction_points(PointDataStructure &attraction_points,
                                                    std::unordered_set<int> &active_attractors,
                                                    const kd_graph_tree_t &tree_node_index,
                                                    real_t kill_distance, real_t influence_radius, int node_count) {
    std::list<int> to_remove;
    std::vector<std::list<int>> node_attractors(node_count, std::list<int>());
    for (const int&  i : active_attractors) {
        std::pair<size_t, real_t> closest_result = get_closest_point<real_t, K, kd_graph_tree_t>(tree_node_index, attraction_points.kdtree_get_pt(i));
        real_t dist = std::sqrt(closest_result.second);
        if (dist < kill_distance) {
            to_remove.push_back(i);
        }
        else if (dist < influence_radius) node_attractors[closest_result.first].push_back(i);
    }
    for (const int& i : to_remove) {
        active_attractors.erase(i);
    }
    return node_attractors;
}

template<typename real_t, int K, typename PointDataStructure>
point_t<real_t, K> calculate_new_node_position(const std::list<int> &attractors,
                                               const PointDataStructure &attraction_points,
                                               const point_t<real_t, K> node_position, real_t step_size) {
    int attractor_count = attractors.size();
    //direction vector
    std::array<real_t, K> dir; for (int i = 0; i < K; i++) dir[i] = 0;
    for (const int& i : attractors) {
        std::array<real_t, K> difference = axpy<real_t, K>(-1, node_position, attraction_points.kdtree_get_pt(i));
        dir = axpy<real_t, K>(1, normalize_vector<real_t, K>(difference), dir);
    }
    dir = normalize_vector<real_t, K>(dir);
    return axpy<real_t, K>(step_size, dir, node_position);
}

//space colonization algorithm
template<typename real_t, int K>
void colonize(PointGraph<real_t, K> &tree_nodes /*in out, with root nodes added*/,
              PointCloud<real_t, K> attraction_points, /*in*/
              real_t step_size, real_t kill_distance, real_t influence_radius,
              int snapshot_every, const std::string &snapfn, int maxiter, int children_limit = 2) {
    
    using kd_graph_tree_t = nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Simple_Adaptor<real_t, PointGraph<real_t, K>>, PointGraph<real_t, K>, K>;

    std::unordered_set<int> active_attractors;
    for (int i = 0; i < attraction_points.kdtree_get_point_count(); i++) active_attractors.insert(i);

    kd_graph_tree_t tree_node_index(K, tree_nodes, {10});

    //remove attraction points close to root nodes and find node attractors
    std::vector<std::list<int>> node_attractors = prune_attraction_points<real_t, K, PointCloud<real_t, K>, kd_graph_tree_t>(
                                                                                        attraction_points,
                                                                                        active_attractors,
                                                                                        tree_node_index,
                                                                                        kill_distance, influence_radius, tree_nodes.kdtree_get_point_count());
    for (int i = 0; i < maxiter; i++) {
        if (i % snapshot_every == 0) tree_nodes.to_file(snapfn + "-" + std::to_string(i / snapshot_every) + ".txt");
        if (active_attractors.empty()) break;
        bool any_new_nodes = false;
        int node_count = tree_nodes.kdtree_get_point_count();
        for (int j = 0; j < node_count; j++) {
            if (!node_attractors[j].empty() && tree_nodes.data_nodes[j].children.size() < children_limit) { //add new nodes
                any_new_nodes = true;
                point_t<real_t, K> new_node_pos = calculate_new_node_position<real_t, K, PointCloud<real_t, K>>(node_attractors[j],
                                                                                                                attraction_points,
                                                                                                                tree_nodes.kdtree_get_pt(j), step_size);
                int new_id = tree_nodes.add_node(j, new_node_pos);
                tree_node_index.addPoints(new_id, new_id);
            }
        }
        if (!any_new_nodes) break;
        //remove attraction points close to root nodes and find node attractors
        node_attractors = prune_attraction_points<real_t, K, PointCloud<real_t, K>, kd_graph_tree_t>(
                                                                                        attraction_points,
                                                                                        active_attractors,
                                                                                        tree_node_index,
                                                                                        kill_distance, influence_radius, tree_nodes.kdtree_get_point_count());
    }
}