DB_CONFIG = {
    'host': 'aws-0-ap-south-1.pooler.supabase.com',
    'database': 'postgres',
    'user': 'postgres.conrxbcvuogbzfysomov',
    'password': 'wXAryCC8@iwNvj#',
    'port': '6543'
}

VALIDATION_THRESHOLDS = {
    'me_consumption': {
        'min': 0,
        'max': 50,
        'power_factor': 250,
        'container_max': 300,
        'non_container_max': 50,
        'historical_lower': 0.8,
        'historical_upper': 1.2,
        'expected_lower': 0.8,
        'expected_upper': 1.2
    }
}

COLUMN_NAMES = {
    'ME_CONSUMPTION': 'actual_me_consumption',
    'ME_POWER': 'actual_me_power',
    'ME_RPM': 'me_rpm',
    'VESSEL_IMO': 'vessel_imo',
    'RUN_HOURS': 'steaming_time_hrs',
    'CURRENT_LOAD': 'me_load_pct',
    'CURRENT_SPEED': 'observed_speed',
    'STREAMING_HOURS': 'steaming_time_hrs',
    'REPORT_DATE': 'reportdate',
    'LOAD_TYPE': 'load_type',
    'VESSEL_NAME': 'vessel_name',
    'VESSEL_TYPE': 'vessel_type',
    'DISPLACEMENT': 'displacement',
    'HULL_PERFORMANCE': 'hull_rough_power_loss_pct_ed'
}
