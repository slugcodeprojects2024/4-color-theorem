"""Build adjacency graph from detected regions."""
import networkx as nx
from typing import Dict, Set, List
import logging

logger = logging.getLogger(__name__)

class GraphBuilder:
    def build_graph(self, adjacency: Dict[int, Set[int]]) -> nx.Graph:
        """
        Build a NetworkX graph from region adjacency information.
        
        Args:
            adjacency: Dictionary mapping region_id to set of adjacent region_ids
            
        Returns:
            NetworkX graph representing region adjacency
        """
        G = nx.Graph()
        
        # Add nodes with attributes
        for region_id in adjacency.keys():
            G.add_node(region_id, region_id=region_id)
        
        # Add edges
        edges_added = set()
        for region_id, neighbors in adjacency.items():
            for neighbor_id in neighbors:
                edge = tuple(sorted([region_id, neighbor_id]))
                if edge not in edges_added:
                    G.add_edge(region_id, neighbor_id)
                    edges_added.add(edge)
        
        # Log graph statistics
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Verify graph is valid for 4-coloring
        if G.number_of_nodes() > 0:
            if nx.is_planar(G):
                logger.info("Graph is planar - 4 colors guaranteed to be sufficient")
            else:
                logger.warning("Graph is not planar - may need more than 4 colors")
        
        return G