import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from shapely.errors import TopologicalError
import logging
from typing import List, Dict, Tuple, Any
import json

logger = logging.getLogger(__name__)

class ZoneDefinitionError(Exception):
    """Custom exception for zone definition errors"""
    pass

class ZoneAnalyzer:
    """
    Parking Zone Violation Analysis System
    
    BUSINESS RULES:
    1. VIP zones → only cars/motorcycles allowed
    2. Standard zones → all vehicles allowed  
    3. Handicap zones → special authorization required
    4. No parking zones → no vehicles allowed
    
    VALIDATION: Point-in-polygon detection with error handling
    """
    
    def __init__(self):
        """Initialize zone analyzer with business rules"""
        
        # Define allowed vehicles per zone type
        self.zone_rules = {
            'vip': ['car', 'motorcycle'],
            'standard': ['car', 'motorcycle', 'bus', 'truck'],
            'handicap': ['car'],  # Requires special validation
            'no_parking': [],     # No vehicles allowed
            'loading': ['truck', 'bus'],  # Commercial vehicles only
        }
        
        # Violation severity levels
        self.violation_severity = {
            'vip_violation': 'high',
            'handicap_violation': 'critical',
            'no_parking_violation': 'medium',
            'loading_zone_violation': 'medium',
            'unauthorized_vehicle': 'low'
        }
        
        logger.info("Zone analyzer initialized with business rules")
        
    def normalize_zone_type(self, zone_type: str) -> str:
        """Normalize zone type for internal comparison"""
        return zone_type.strip().replace("-", "_").replace(" ", "_").lower()

    
    def validate_zone_definition(self, zone: Dict[str, Any]) -> bool:
        required_fields = ['id', 'type', 'points']

        # Check required fields
        for field in required_fields:
            if field not in zone:
                raise ZoneDefinitionError(f"Missing required field: {field}")

        # Normalize and debug log
        raw_type = zone['type']
        zone_type = self.normalize_zone_type(raw_type)
        print(f"[DEBUG] Zone ID: {zone.get('id')} | Raw type: '{raw_type}' | Normalized: '{zone_type}'")

        if zone_type not in self.zone_rules:
            raise ZoneDefinitionError(f"Invalid zone type: {raw_type}")

        # Replace with normalized type
        zone['type'] = zone_type

        # Validate points (must have at least 3 points for polygon)
        points = zone['points']
        if not isinstance(points, list) or len(points) < 3:
            raise ZoneDefinitionError("Zone must have at least 3 points")

        for i, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ZoneDefinitionError(f"Point {i} must be [x, y] coordinates")
            try:
                float(point[0])
                float(point[1])
            except (ValueError, TypeError):
                raise ZoneDefinitionError(f"Point {i} coordinates must be numeric")

        return True

    
    def create_polygon_from_zone(self, zone: Dict[str, Any]) -> Polygon:
        """
        Create Shapely polygon from zone definition
        
        Args:
            zone (dict): Zone definition with points
            
        Returns:
            Polygon: Shapely polygon object
            
        Raises:
            ZoneDefinitionError: If polygon creation fails
        """
        try:
            points = [(float(p[0]), float(p[1])) for p in zone['points']]
            polygon = Polygon(points)
            
            if not polygon.is_valid:
                # Try to fix invalid polygons
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    raise ZoneDefinitionError(f"Cannot create valid polygon for zone {zone['id']}")
            
            return polygon
            
        except Exception as e:
            raise ZoneDefinitionError(f"Polygon creation failed for zone {zone['id']}: {e}")
    
    def calculate_vehicle_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Calculate vehicle centroid from bounding box
        
        Args:
            bbox (list): Bounding box [x1, y1, x2, y2]
            
        Returns:
            tuple: (x, y) centroid coordinates
            
        BUSINESS RULE: Use bottom-center of bbox for better ground contact
        """
        x1, y1, x2, y2 = bbox
        
        # Use bottom-center point (better represents vehicle ground contact)
        centroid_x = (x1 + x2) / 2
        centroid_y = y2  # Bottom of bounding box
        
        return (centroid_x, centroid_y)
    
    def is_point_in_zone(self, point: Tuple[float, float], polygon: Polygon) -> bool:
        """
        Check if point is inside polygon using robust geometry
        
        Args:
            point (tuple): (x, y) coordinates
            polygon (Polygon): Zone polygon
            
        Returns:
            bool: True if point is inside polygon
        """
        try:
            shapely_point = Point(point)
            return polygon.contains(shapely_point) or polygon.touches(shapely_point)
        except TopologicalError:
            # Fallback to buffer method for edge cases
            try:
                return polygon.buffer(1e-10).contains(Point(point))
            except:
                logger.warning(f"Geometry check failed for point {point}")
                return False
    
    def check_vehicle_authorization(self, vehicle_type: str, zone_type: str, zone_metadata: Dict = None) -> Dict[str, Any]:
        """
        Check if vehicle is authorized in zone type
        
        Args:
            vehicle_type (str): Type of vehicle ('car', 'truck', etc.)
            zone_type (str): Type of zone ('vip', 'standard', etc.)
            zone_metadata (dict): Additional zone information
            
        Returns:
            dict: Authorization result with details
        """
        result = {
            'authorized': False,
            'violation_type': None,
            'severity': 'low',
            'message': ''
        }
        
        allowed_vehicles = self.zone_rules.get(zone_type, [])
        
        if vehicle_type in allowed_vehicles:
            result['authorized'] = True
            result['message'] = f"{vehicle_type} authorized in {zone_type} zone"
        else:
            result['authorized'] = False
            
            # Determine specific violation type
            if zone_type == 'vip':
                result['violation_type'] = 'vip_violation'
                result['message'] = f"{vehicle_type} not allowed in VIP zone (cars/motorcycles only)"
            elif zone_type == 'handicap':
                result['violation_type'] = 'handicap_violation'
                result['message'] = f"{vehicle_type} in handicap zone without authorization"
            elif zone_type == 'no_parking':
                result['violation_type'] = 'no_parking_violation'
                result['message'] = f"{vehicle_type} parked in no-parking zone"
            elif zone_type == 'loading':
                result['violation_type'] = 'loading_zone_violation'
                result['message'] = f"{vehicle_type} not allowed in loading zone (commercial vehicles only)"
            else:
                result['violation_type'] = 'unauthorized_vehicle'
                result['message'] = f"{vehicle_type} not authorized in {zone_type} zone"
            
            result['severity'] = self.violation_severity.get(result['violation_type'], 'low')
        
        return result
    
    def check_violations(self, zones: List[Dict], detections: List[Dict]) -> List[Dict]:
        """
        Check for parking violations across all zones and detections
        
        Args:
            zones (list): List of zone definitions
            detections (list): List of vehicle detections
            
        Returns:
            list: List of violation records
            
        BUSINESS RULE: Each violation must be documented with zone, vehicle, and violation type
        """
        violations = []
        
        if not zones or not detections:
            logger.info("No zones or detections provided for violation checking")
            return violations
        
        try:
            # Validate and create polygons for all zones
            zone_polygons = {}
            for zone in zones:
                try:
                    self.validate_zone_definition(zone)
                    zone_polygons[zone['id']] = {
                        'polygon': self.create_polygon_from_zone(zone),
                        'type': zone['type'], 
                        'metadata': zone.get('metadata', {})

                    }
                except ZoneDefinitionError as e:
                    logger.error(f"Invalid zone {zone.get('id', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Processing {len(detections)} detections against {len(zone_polygons)} valid zones")
            
            # Check each detection against all zones
            for detection_idx, detection in enumerate(detections):
                try:
                    vehicle_centroid = self.calculate_vehicle_centroid(detection['bbox'])
                    vehicle_type = detection.get('cls_name', 'unknown')
                    
                    # Check against each zone
                    for zone_id, zone_data in zone_polygons.items():
                        if self.is_point_in_zone(vehicle_centroid, zone_data['polygon']):
                            # Vehicle is in this zone - check authorization
                            auth_result = self.check_vehicle_authorization(
                                vehicle_type, 
                                zone_data['type'],
                                zone_data['metadata']
                            )
                            
                            if not auth_result['authorized']:
                                violation = {
                                    'id': f"violation_{len(violations) + 1}",
                                    'zone_id': zone_id,
                                    'zone_type': zone_data['type'],
                                    'vehicle_detection_idx': detection_idx,
                                    'vehicle_type': vehicle_type,
                                    'vehicle_bbox': detection['bbox'],
                                    'vehicle_centroid': vehicle_centroid,
                                    'vehicle_confidence': detection.get('conf', 0.0),
                                    'violation_type': auth_result['violation_type'],
                                    'severity': auth_result['severity'],
                                    'message': auth_result['message'],
                                    'timestamp': None 
                                }
                                violations.append(violation)
                                
                                logger.info(f"Violation detected: {violation['message']}")
                
                except Exception as e:
                    logger.error(f"Error processing detection {detection_idx}: {e}")
                    continue
            
            logger.info(f"Found {len(violations)} total violations")
            return violations
            
        except Exception as e:
            logger.error(f"Violation checking failed: {e}")
            return []
    
    def generate_violation_report(self, violations: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive violation report with statistics
        
        Args:
            violations (list): List of violation records
            
        Returns:
            dict: Detailed violation report
        """
        if not violations:
            return {
                'total_violations': 0,
                'by_severity': {},
                'by_zone_type': {},
                'by_vehicle_type': {},
                'recommendations': ["No violations detected. All vehicles are properly parked."]
            }
        
        report = {
            'total_violations': len(violations),
            'by_severity': {},
            'by_zone_type': {},
            'by_vehicle_type': {},
            'critical_violations': [],
            'recommendations': []
        }
        
        # Analyze violations by different categories
        for violation in violations:
            severity = violation['severity']
            zone_type = violation['zone_type']
            vehicle_type = violation['vehicle_type']
            
            # Count by severity
            report['by_severity'][severity] = report['by_severity'].get(severity, 0) + 1
            
            # Count by zone type
            report['by_zone_type'][zone_type] = report['by_zone_type'].get(zone_type, 0) + 1
            
            # Count by vehicle type
            report['by_vehicle_type'][vehicle_type] = report['by_vehicle_type'].get(vehicle_type, 0) + 1
            
            # Track critical violations
            if severity == 'critical':
                report['critical_violations'].append(violation)
        
        # Generate recommendations
        if report['by_severity'].get('critical', 0) > 0:
            report['recommendations'].append("URGENT: Critical violations detected in handicap zones. Immediate action required.")
        
        if report['by_severity'].get('high', 0) > 0:
            report['recommendations'].append("High-priority violations in VIP zones. Consider enforcement action.")
        
        if report['by_zone_type'].get('no_parking', 0) > 0:
            report['recommendations'].append("Vehicles detected in no-parking zones. Review signage and enforcement.")
        
        # Zone-specific recommendations
        most_violated_zone = max(report['by_zone_type'].items(), key=lambda x: x[1])[0]
        report['recommendations'].append(f"Most violations in {most_violated_zone} zones. Consider additional monitoring.")
        
        return report
    
    def calculate_occupancy_metrics(self, zones: List[Dict], detections: List[Dict]) -> Dict[str, Any]:
        """
        Calculate detailed occupancy metrics per zone type
        
        Args:
            zones (list): List of zone definitions
            detections (list): List of vehicle detections
            
        Returns:
            dict: Occupancy metrics by zone type
        """
        metrics = {
            'total_spots': 0,
            'occupied_spots': 0,
            'occupancy_rate': 0.0,
            'by_zone_type': {}
        }
        
        try:
            # Initialize zone type counters
            zone_types = {}
            for zone in zones:
                zone_type = zone.get('type', 'unknown')
                if zone_type not in zone_types:
                    zone_types[zone_type] = {'total': 0, 'occupied': 0, 'zones': []}
                zone_types[zone_type]['total'] += 1
                zone_types[zone_type]['zones'].append(zone)
            
            # Count occupied spots
            zone_polygons = {}
            for zone in zones:
                if 'id' in zone and 'points' in zone:
                    try:
                        zone_polygons[zone['id']] = {
                            'polygon': self.create_polygon_from_zone(zone),
                            'type': zone.get('type', 'unknown'),
                            'occupied': False
                        }
                    except:
                        continue
            
            # Check which zones have vehicles
            for detection in detections:
                try:
                    vehicle_centroid = self.calculate_vehicle_centroid(detection['bbox'])
                    
                    for zone_id, zone_data in zone_polygons.items():
                        if self.is_point_in_zone(vehicle_centroid, zone_data['polygon']):
                            if not zone_data['occupied']:  # Only count once per zone
                                zone_data['occupied'] = True
                                zone_types[zone_data['type']]['occupied'] += 1
                                break
                except:
                    continue
            
            # Calculate metrics
            metrics['total_spots'] = len(zones)
            metrics['occupied_spots'] = sum(1 for z in zone_polygons.values() if z['occupied'])
            
            if metrics['total_spots'] > 0:
                metrics['occupancy_rate'] = round((metrics['occupied_spots'] / metrics['total_spots']) * 100, 1)
            
            # Per zone type metrics
            for zone_type, data in zone_types.items():
                if data['total'] > 0:
                    occupancy_rate = round((data['occupied'] / data['total']) * 100, 1)
                    metrics['by_zone_type'][zone_type] = {
                        'total_spots': data['total'],
                        'occupied_spots': data['occupied'],
                        'occupancy_rate': occupancy_rate,
                        'available_spots': data['total'] - data['occupied']
                    }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Occupancy calculation failed: {e}")
            return metrics

def calculate_occupancy(detections: List[Dict], total_spots: int) -> float:
    """
    Calculate parking lot occupancy percentage
    
    BUSINESS RULE: Only count vehicles in designated spots
    VALIDATION: Ensure count <= total spots
    
    Args:
        detections (list): Vehicle detection objects
        total_spots (int): Total parking capacity
    
    Returns:
        float: Occupancy percentage (0-100)
    
    Edge Cases:
        - More detections than spots → cap at 100%
        - Empty lot → return 0.0
    """
    if total_spots <= 0:
        logger.warning("Invalid total_spots value")
        return 0.0
    
    if not detections:
        return 0.0
    
    # Count valid vehicle detections
    vehicle_count = len([d for d in detections if d.get('cls_name') in ['car', 'motorcycle', 'bus', 'truck']])
    
    # Cap at 100% occupancy
    occupied_spots = min(vehicle_count, total_spots)
    occupancy = (occupied_spots / total_spots) * 100
    
    return round(occupancy, 1)
