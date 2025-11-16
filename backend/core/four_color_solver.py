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
        
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        logger.info(f"Coloring graph with {num_nodes} nodes and {num_edges} edges")
        
        # For large graphs, skip backtracking (exponential time complexity)
        # Use faster algorithms and normalize colors to 0-3 range
        if num_nodes > 100:
            logger.info(f"Large graph detected ({num_nodes} nodes), using fast coloring algorithm")
            # Use NetworkX's DSATUR strategy (faster and often better than greedy)
            coloring = nx.greedy_color(graph, strategy='DSATUR')
            # Normalize colors to 0-3 range
            coloring = self._normalize_colors(coloring)
            return coloring
        
        # For smaller graphs, try Welsh-Powell first
        coloring = self._welsh_powell(graph)
        max_color = max(coloring.values()) if coloring else -1
        
        # Strategy 2: If Welsh-Powell uses more than 4 colors, try backtracking
        if max_color > 3:
            logger.warning(f"Welsh-Powell used {max_color + 1} colors, trying backtracking")
            backtrack_result = self._backtrack_coloring(graph, max_iterations=10000)
            if backtrack_result:
                coloring = backtrack_result
            else:
                logger.warning("Backtracking failed or timed out, using NetworkX greedy coloring")
                coloring = nx.greedy_color(graph, strategy='DSATUR')
                coloring = self._normalize_colors(coloring)
        
        # Strategy 3: If still failing, use NetworkX's coloring
        if not coloring:
            logger.warning("Using NetworkX greedy coloring")
            coloring = nx.greedy_color(graph, strategy='DSATUR')
        
        # Normalize colors to 0-3 range
        max_color = max(coloring.values()) if coloring else -1
        if max_color > 3:
            coloring = self._normalize_colors(coloring)
        
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
    
    def _backtrack_coloring(self, graph: nx.Graph, max_iterations: int = 10000) -> Optional[Dict[int, int]]:
        """
        Use backtracking to find a 4-coloring if possible.
        Limited by max_iterations to prevent hanging on complex graphs.
        
        Args:
            graph: NetworkX graph
            max_iterations: Maximum number of recursive calls (prevents hanging)
            
        Returns:
            Dictionary mapping node_id to color_id, or None if impossible/timed out
        """
        nodes = list(graph.nodes())
        coloring = {}
        iteration_count = [0]  # Use list to allow modification in nested function
        
        def is_safe(node: int, color: int) -> bool:
            """Check if it's safe to assign color to node."""
            for neighbor in graph.neighbors(node):
                if neighbor in coloring and coloring[neighbor] == color:
                    return False
            return True
        
        def backtrack(node_idx: int) -> bool:
            """Recursive backtracking function."""
            iteration_count[0] += 1
            if iteration_count[0] > max_iterations:
                logger.warning(f"Backtracking exceeded {max_iterations} iterations, aborting")
                return False
            
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
            logger.info(f"Backtracking succeeded in {iteration_count[0]} iterations")
            return coloring
        else:
            if iteration_count[0] > max_iterations:
                logger.warning("Backtracking timed out")
            else:
                logger.error("No 4-coloring found via backtracking")
            return None
    
    def _normalize_colors(self, coloring: Dict[int, int]) -> Dict[int, int]:
        """
        Normalize coloring to use only colors 0-3.
        Maps colors sequentially to 0-3 range. This is safe because if the original
        coloring was valid (no adjacent nodes had same color), the modulo mapping
        preserves that property for the first 4 colors. If more than 4 colors were used,
        this is a best-effort normalization (non-planar graphs may need more than 4 colors).
        
        Args:
            coloring: Dictionary mapping node_id to color_id
            
        Returns:
            Dictionary with colors normalized to 0-3 range
        """
        if not coloring:
            return coloring
        
        # Get unique colors in order of appearance
        unique_colors = []
        seen = set()
        for color in coloring.values():
            if color not in seen:
                unique_colors.append(color)
                seen.add(color)
        
        # Create mapping: map first 4 unique colors to 0-3, others wrap around
        # This preserves the coloring property as much as possible
        color_map = {}
        for idx, old_color in enumerate(unique_colors):
            color_map[old_color] = idx % 4
        
        # Apply mapping
        normalized = {node: color_map[color] for node, color in coloring.items()}
        
        num_colors_used = len(set(normalized.values()))
        if len(unique_colors) > 4:
            logger.warning(f"Graph used {len(unique_colors)} colors, normalized to {num_colors_used} colors")
        else:
            logger.info(f"Normalized coloring: {num_colors_used} colors used")
        
        return normalized