import numpy as np

# Density thresholds (people per m²)
SAFE = 2
WARNING = 4
DANGER = 7

def split_into_zones(density_map, grid=(3, 3)):
    
    h, w = density_map.shape
    rows, cols = grid

    zone_counts = []

    row_size = h // rows
    col_size = w // cols

    for r in range(rows):
        for c in range(cols):
            # Extract zone patch
            zone = density_map[
                r * row_size : (r + 1) * row_size,
                c * col_size : (c + 1) * col_size
            ]
            zone_counts.append(zone.sum())

    return zone_counts  # list of 9 values

def get_zone_status(zone_count, zone_area_m2=10.0):
    
    density = zone_count / zone_area_m2

    if density < SAFE:
        return density, "SAFE", "green"
    elif density < WARNING:
        return density, "WARNING", "yellow"
    elif density < DANGER:
        return density, "DANGER", "orange"
    else:
        return density, "CRITICAL", "red"

def get_all_zone_alerts(density_map, grid=(3, 3), zone_area=10.0):
    zone_counts = split_into_zones(density_map, grid)
    alerts = []
    for i, count in enumerate(zone_counts):
        density, status, color = get_zone_status(count, zone_area_m2=zone_area)
        alerts.append({
            "zone": f"Zone {i+1}",
            "count": int(count),
            "density": round(float(density), 1) if float(density) < 10 else int(round(float(density))),
            "status": status,
            "color": color
        })
    return alerts

def get_overall_risk(alerts):
    
    statuses = [a["status"] for a in alerts]

    if "CRITICAL" in statuses:
        return "CRITICAL"
    elif "DANGER" in statuses:
        return "DANGER"
    elif "WARNING" in statuses:
        return "WARNING"
    else:
        return "SAFE"