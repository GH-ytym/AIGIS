"""OSMnx graph loader for local network-based analysis."""

from __future__ import annotations

import networkx as nx
import osmnx as ox

from aigis_agent.schemas.routing import Coordinate


class GraphService:
    """Build and access local graph snapshots from OSM data."""

    def load_drive_graph(self, center: Coordinate, dist_m: int = 2000) -> nx.MultiDiGraph:
        """Load a drive network around a center point."""
        # TODO: persist/download cache to avoid repeated OSM requests.
        return ox.graph_from_point((center.lat, center.lon), dist=dist_m, network_type="drive")
