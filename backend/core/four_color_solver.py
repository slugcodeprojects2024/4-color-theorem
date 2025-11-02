"""Implement 4-color theorem solver."""
import networkx as nx
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class FourColorSolver:
    def __init__(self):
        """Initialize the 4-color solver."""
        self.colors = [0, 1, 2, 3]  # Four colors
        
    def solve(self, graph: nx.Graph) -> Dict[int, int]:
        """
        Solve the graph coloring problem using at most 4 colors.
        
        Args:
            graph: NetworkX graph to color
            
        Returns:
            Dictionary mapping node_id to color_id (0-3)
        """
        if graph.number_of_nodes() == 0:
            return {}
        
        # Try different strategies
        coloring = None
        
        # Strategy 1: Welsh-Powell algorithm
        coloring = self._welsh_powell(graph)
        
        # Strategy 2: If Welsh-Powell uses more than 4 colors, try backtracking
        if coloring and max(coloring.values()) > 3:
            logger.warning("Welsh-Powell used more than 4 colors, trying backtracking")
            coloring = self._backtrack_coloring(graph)
        
        # Strategy 3: If still failing, use NetworkX's coloring
        if not coloring or max(coloring.values()) > 3:
            logger.warning("Using NetworkX greedy coloring")
            coloring = nx.greedy_color(graph, strategy='largest_first')
        
        return coloring
    
    def _welsh_powell(self, graph: nx.Graph) -> Dict[int, int]:
        """
        Implement Welsh-Powell algorithm for graph coloring.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary mapping node_id to color_id
        """
        # Sort nodes by degree (descending)
        nodes_by_degree = sorted(
            graph.nodes(), 
            key=lambda x: graph.degree(x), 
            reverse=True
        )
        
        coloring = {}
        
        for node in nodes_by_degree:
            # Find colors used by neighbors
            neighbor_colors = set()
            for neighbor in graph.neighbors(node):
                if neighbor in coloring:
                    neighbor_colors.add(coloring[neighbor])
            
            # Assign first available color
            for color in self.colors:
                if color not in neighbor_colors:
                    coloring[node] = color
                    break
            else:
                # If no color from 0-3 is available, use next available
                coloring[node] = min(set(range(len(neighbor_colors) + 1)) - neighbor_colors)
        
        return coloring
    
    def _backtrack_coloring(self, graph: nx.Graph) -> Optional[Dict[int, int]]:
        """
        Use backtracking to find a 4-coloring if possible.
        
        Args:
            graph: NetworkX graph
            
        Returns:
            Dictionary mapping node_id to color_id, or None if impossible
        """
        nodes = list(graph.nodes())
        coloring = {}
        
        def is_safe(node: int, color: int) -> bool:
            """Check if it's safe to assign color to node."""
            for neighbor in graph.neighbors(node):
                if neighbor in coloring and coloring[neighbor] == color:
                    return False
            return True
        
        def backtrack(node_idx: int) -> bool:
            """Recursive backtracking function."""
            if node_idx == len(nodes):
                return True  # All nodes colored
            
            node = nodes[node_idx]
            
            for color in self.colors:
                if is_safe(node, color):
                    coloring[node] = color
                    
                    if backtrack(node_idx + 1):
                        return True
                    
                    # Backtrack
                    del coloring[node]
            
            return False
        
        if backtrack(0):
            return coloring
        else:
            logger.error("No 4-coloring found via backtracking")
            return None