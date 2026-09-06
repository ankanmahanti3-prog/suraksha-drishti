# Synthetic Authorised Identity & HRMS Registry (Prototype Scope)
PERSONNEL_REGISTRY = {
    "P-10001": {
        "pin": "1234",
        "display_id": "P-10001 (Demo)",
        "role": "Personnel",
        "unit": "Alpha Company",
        "status": "Active",
        "days_since_leave": 82,
        "night_shifts_7d": 4,
        "deployment_hardship": "Counter-Insurgency / High Hardship",
        "trauma_incident_flag": True,
        "sleep_hours": 4.0,
        "subjective_fatigue": 3
    },
    "P-10002": {
        "pin": "1234",
        "display_id": "P-10002 (Demo)",
        "role": "Personnel",
        "unit": "Alpha Company",
        "status": "Active",
        "days_since_leave": 18,
        "night_shifts_7d": 0,
        "deployment_hardship": "Peace Station / Standard Base",
        "trauma_incident_flag": False,
        "sleep_hours": 7.5,
        "subjective_fatigue": 1
    },
    "C-20001": {
        "pin": "2345",
        "display_id": "C-20001 (Demo)",
        "role": "Commander",
        "unit": "Alpha Company",
        "status": "Active"
    },
    "W-30001": {
        "pin": "3456",
        "display_id": "W-30001 (Demo)",
        "role": "Welfare Officer",
        "unit": "Battalion HQ / Medical",
        "status": "Active"
    }
}

RBAC_PERMISSIONS = {
    "Personnel": ["view_own_wellness", "submit_self_assessment", "optional_telemetry"],
    "Commander": ["view_workload_grid", "view_unit_fatigue_distribution", "generate_roster_recommendation"],
    "Welfare Officer": ["view_longitudinal_trends", "view_xai_attribution", "initiate_welfare_actions", "view_confidential_cases"]
}

def authenticate_user(service_id, pin):
    user = PERSONNEL_REGISTRY.get(service_id)
    if not user:
        return False, "Access Denied: Service ID not found in authorised personnel registry."
    if user["status"] != "Active":
        return False, "Access Denied: Account status is inactive or suspended."
    if user["pin"] != pin:
        return False, "Access Denied: Invalid authentication credential."
    return True, user

def check_permission(role, permission):
    return permission in RBAC_PERMISSIONS.get(role, [])