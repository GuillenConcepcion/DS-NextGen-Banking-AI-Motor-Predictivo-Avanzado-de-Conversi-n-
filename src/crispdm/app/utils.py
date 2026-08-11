import numpy as np
import math
import plotly.graph_objects as go
from typing import List, Tuple, Optional

def generate_levels(num_levels: int, min_radius: float = 0.0, max_radius: float = 1.0) -> List[float]:
    """Generates equally spaced radial levels for the parliament chart."""
    return np.linspace(min_radius, max_radius, num_levels).tolist()

def generate_points(levels: List[float], total_points: int) -> List[int]:
    """Calculates the number of points proportional to the circumference of each level."""
    circumferences = [2 * math.pi * r for r in levels]
    total_circumference = sum(circumferences)
    
    if total_circumference == 0:
        return [0] * len(levels)
        
    proportional_points = [(c / total_circumference) * total_points for c in circumferences]
    
    points_per_level = [int(p) for p in proportional_points]
    residuals = [p - int(p) for p in proportional_points]
    
    while sum(points_per_level) < total_points:
        max_index = residuals.index(max(residuals))
        points_per_level[max_index] += 1
        residuals[max_index] = 0

    return points_per_level

def generate_radii_theta(levels: List[float], points_per_level: List[int], theta_start: float, theta_end: float) -> Tuple[List[float], List[float]]:
    """Generates coordinates (radius and theta) for the points distributed in the levels."""
    radii = []
    theta = []
    for level, count in zip(levels, points_per_level):
        if count > 0:
            level_theta_values = np.linspace(theta_start, theta_end, count, endpoint=True).tolist()
            radii.extend([level] * count)
            theta.extend(level_theta_values)

    # Sort coordinates primarily by theta, then by radius
    radii_theta_sorted = sorted(zip(radii, theta), key=lambda x: (x[1], x[0]))
    if not radii_theta_sorted:
        return [], []
        
    radii_sorted, theta_sorted = zip(*radii_theta_sorted)
    return list(radii_sorted), list(theta_sorted)

def create_parliament_chart(parties: List[str], party_counts: List[int], colors: List[str], radii_sorted: List[float], theta_sorted: List[float], marker_size: int = 10) -> go.Figure:
    """Creates a parliament-style semi-circle chart using polar coordinates."""
    fig = go.Figure()
    party_start_idx = 0
    for i, party in enumerate(parties):
        party_end_idx = party_start_idx + party_counts[i]
        fig.add_trace(go.Scatterpolar(
            r=radii_sorted[party_start_idx:party_end_idx],
            theta=theta_sorted[party_start_idx:party_end_idx],
            mode='markers',
            marker=dict(size=marker_size, color=colors[i]),
            name=party,
            legendgroup=party,
            hovertemplate=f"Group: {party}<br>Points: {party_counts[i]}<extra></extra>"
        ))
        party_start_idx = party_end_idx
    return fig

def setup_layout(fig: go.Figure, title: str, subtitle: str, legend_orientation: Optional[str] = None) -> None:
    """Configures the aesthetic layout for the Plotly polar chart."""
    annotation_position = -0.0885 if legend_orientation == 'h' else -0.1015
    layout = dict(
        title=title,
        showlegend=True,
        polar=dict(
            radialaxis=dict(showline=False, showticklabels=False, linecolor='#0e1117', gridcolor="#0e1117"),
            angularaxis=dict(showline=False, showticklabels=False, linecolor='#0e1117', gridcolor="#0e1117"),
            bgcolor='#0e1117'
        ),
        height=600, width=700,
        font=dict(family="Poppins, sans-serif"),
        annotations=[dict(
            x=annotation_position, y=1.1, xref='paper', yref='paper',
            text=subtitle, showarrow=False,
            font=dict(size=14, color='grey'), xanchor='left'
        )]
    )
    if legend_orientation:
        layout['legend'] = dict(
            orientation=legend_orientation, yanchor='bottom',
            y=0.1 if legend_orientation == 'h' else 1,
            xanchor='center', x=0.5
        )
    fig.update_layout(**layout)
