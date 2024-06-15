#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

# Define a structure to represent a node in the TSP problem
struct Node {
    double x;
    double y;
};

# Function to calculate the Euclidean distance between two nodes
double calculateDistance(const Node& node1, const Node& node2) {
    return std::sqrt(std::pow(node1.x - node2.x, 2) + std::pow(node1.y - node2.y, 2));
}

# Function to generate a random TSP instance
std::vector<Node> generateTSPInstance(int numNodes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);

    std::vector<Node> nodes(numNodes);
    for (int i = 0; i < numNodes; ++i) {
        nodes[i].x = dist(gen);
        nodes[i].y = dist(gen);
    }

    return nodes;
}

# Function to calculate the total distance of a given tour
double calculateTourDistance(const std::vector<Node>& nodes, const std::vector<int>& tour) {
    double totalDistance = 0.0;
    for (int i = 0; i < tour.size() - 1; ++i) {
        totalDistance += calculateDistance(nodes[tour[i]], nodes[tour[i + 1]]);
    }
    totalDistance += calculateDistance(nodes[tour.back()], nodes[tour.front()]);
    return totalDistance;
}

# Function to implement the nearest neighbor heuristic
std::vector<int> nearestNeighborHeuristic(const std::vector<Node>& nodes) {
    int numNodes = nodes.size();
    std::vector<int> tour(numNodes);
    std::vector<bool> visited(numNodes, false);

    # Start at a random node
    int current = rand() % numNodes;
    tour[0] = current;
    visited[current] = true;

    for (int i = 1; i < numNodes; ++i) {
        double minDistance = std::numeric_limits<double>::max();
        int nearest = -1;

        # Find the nearest unvisited node
        for (int j = 0; j < numNodes; ++j) {
            if (!visited[j]) {
                double distance = calculateDistance(nodes[current], nodes[j]);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearest = j;
                }
            }
        }

        # Add the nearest node to the tour
        tour[i] = nearest;
        visited[nearest] = true;
        current = nearest;
    }

    return tour;
}

int main() {
    # Generate a TSP instance with 10 nodes
    int numNodes = 10;
    std::vector<Node> nodes = generateTSPInstance(numNodes);

    # Solve the TSP using the nearest neighbor heuristic
    std::vector<int> tour = nearestNeighborHeuristic(nodes);

    # Calculate the total distance of the tour
    double tourDistance = calculateTourDistance(nodes, tour);

    # Print the tour and its distance
    std::cout << "Tour: ";
    for (int node : tour) {
        std::cout << node << " ";
    }
    std::cout << std::endl;
    std::cout << "Distance: " << tourDistance << std::endl;

    return 0;
}
