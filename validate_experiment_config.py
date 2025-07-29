import yaml
import sqlite3
import argparse
import os
from typing import Dict, List, Any

class ExperimentConfigValidator:
    def __init__(self, conditioning_db_path: str):
        self.conn = sqlite3.connect(conditioning_db_path)
    
    def validate_config(self, config_path: str) -> Dict[str, Any]:
        """Validate experiment configuration against available conditions."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        report = {"valid": True, "warnings": [], "suggestions": [], "stats": {}}
        
        for cs in config.get('conditioning', {}).get('condition_sets', []):
            cs_id = cs["condition_set_id"]
            
            # Check if condition set exists
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conditions WHERE experiment_id = ?", (cs_id,))
            total_conditions = cursor.fetchone()[0]
            
            if total_conditions == 0:
                report["warnings"].append(f"Condition set '{cs_id}' not found or empty")
                report["valid"] = False
                continue
            
            # Discover available types
            cursor.execute("SELECT DISTINCT type FROM conditions WHERE experiment_id = ?", (cs_id,))
            available_types = [row[0] for row in cursor.fetchall()]
            
            # Group types by base type
            type_families = {}
            for type_name in available_types:
                if '_' in type_name:
                    base_type, variant = type_name.split('_', 1)
                    if base_type not in type_families:
                        type_families[base_type] = []
                    type_families[base_type].append(variant)
                else:
                    type_families[type_name] = ['base']
            
            # Validate requested types
            requested_types = cs.get("types", [])
            for requested_type in requested_types:
                if requested_type not in type_families:
                    # Check if any base type starts with this
                    matches = [base for base in type_families.keys() if base.startswith(requested_type)]
                    if matches:
                        report["suggestions"].append(
                            f"Type '{requested_type}' not found. Did you mean one of: {matches}?"
                        )
                    else:
                        report["warnings"].append(
                            f"Type '{requested_type}' not found in condition set '{cs_id}'"
                        )
            
            # Validate type_variants
            if "type_variants" in cs:
                for base_type, variant_config in cs["type_variants"].items():
                    if base_type not in type_families:
                        report["warnings"].append(
                            f"Base type '{base_type}' not found in condition set '{cs_id}'"
                        )
                    else:
                        variants = variant_config.get("variants", [])
                        if variants != "all":
                            available_variants = type_families[base_type]
                            for variant in variants:
                                if variant not in available_variants and variant != "base":
                                    report["warnings"].append(
                                        f"Variant '{variant}' not available for '{base_type}'. Available: {available_variants}"
                                    )
            
            # Simulate query to count matching conditions
            matching_count = self._simulate_query(cs, cs_id)
            
            report["stats"][cs_id] = {
                "total_available": total_conditions,
                "would_select": matching_count,
                "types_available": list(type_families.keys()),
                "type_families": type_families
            }
        
        return report
    
    def _simulate_query(self, cs_config: Dict, cs_id: str) -> int:
        """Simulate the DMRunner query to count matching conditions."""
        cursor = self.conn.cursor()
        
        # Build query similar to DMRunner
        query_parts = ["SELECT COUNT(*) FROM conditions WHERE experiment_id = ?"]
        params: List[Any] = [cs_id]
        
        # Type conditions
        type_conditions = self._build_type_conditions(cs_config, params)
        if type_conditions:
            query_parts.append(f"AND ({type_conditions})")
        
        # Filter conditions
        filter_conditions = self._build_filter_conditions(cs_config, params)
        if filter_conditions:
            query_parts.append(f"AND ({filter_conditions})")
        
        query = " ".join(query_parts)
        
        try:
            cursor.execute(query, tuple(params))
            return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error simulating query: {e}")
            return 0
    
    def _build_type_conditions(self, cs_config: Dict, params: List[Any]) -> str:
        """Build SQL conditions for type selection (same as DMRunner)."""
        conditions = []
        
        if "types" in cs_config:
            for requested_type in cs_config["types"]:
                conditions.append("type = ? OR type LIKE ?")
                params.extend([requested_type, f"{requested_type}_%"])
        
        if "type_variants" in cs_config:
            for base_type, config in cs_config["type_variants"].items():
                variants = config.get("variants", [])
                
                if variants == "all":
                    conditions.append("type LIKE ?")
                    params.append(f"{base_type}%")
                else:
                    variant_conditions = []
                    for variant in variants:
                        if variant == "base":
                            variant_conditions.append("type = ?")
                            params.append(base_type)
                        else:
                            variant_conditions.append("type = ?")
                            params.append(f"{base_type}_{variant}")
                    
                    if variant_conditions:
                        conditions.append(f"({' OR '.join(variant_conditions)})")
        
        return " OR ".join(conditions) if conditions else ""
    
    def _build_filter_conditions(self, cs_config: Dict, params: List[Any]) -> str:
        """Build SQL conditions for advanced filtering (same as DMRunner)."""
        conditions = []
        filters = cs_config.get("filters", {})
        
        if "objects" in filters:
            placeholders = ','.join('?' for _ in filters["objects"])
            conditions.append(f"object IN ({placeholders})")
            params.extend(filters["objects"])
        
        if "number_range" in filters:
            placeholders = ','.join('?' for _ in filters["number_range"])
            conditions.append(f"number IN ({placeholders})")
            params.extend(filters["number_range"])
        
        if "backgrounds" in filters:
            bg_conditions = []
            for bg in filters["backgrounds"]:
                bg_conditions.append("prompt LIKE ?")
                params.append(f"%{bg}%")
            if bg_conditions:
                conditions.append(f"({' OR '.join(bg_conditions)})")
        
        if "custom_where" in filters:
            conditions.append(filters["custom_where"])
        
        return " AND ".join(conditions) if conditions else ""

def main():
    parser = argparse.ArgumentParser(description="Validate DMCAF experiment configuration")
    parser.add_argument("config_path", help="Path to experiment config YAML file")
    parser.add_argument("--conditioning_db", help="Path to conditioning database", 
                       default=None)
    
    args = parser.parse_args()
    
    # Get conditioning DB path
    if args.conditioning_db:
        conditioning_db_path = args.conditioning_db
    else:
        data_dir = os.getenv('OUTPUT_DIRECTORY')
        if not data_dir:
            print("ERROR: OUTPUT_DIRECTORY environment variable not set")
            return
        
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        conditioning_db_path = os.path.join(data_dir, config['conditioning_db'])
    
    if not os.path.exists(conditioning_db_path):
        print(f"ERROR: Conditioning database not found: {conditioning_db_path}")
        return
    
    # Validate configuration
    validator = ExperimentConfigValidator(conditioning_db_path)
    report = validator.validate_config(args.config_path)
    
    # Print report
    print("=" * 60)
    print("DMCAF EXPERIMENT CONFIG VALIDATION REPORT")
    print("=" * 60)
    
    if report["valid"] and not report["warnings"]:
        print("✅ Configuration is valid!")
    else:
        print("⚠️  Configuration has issues:")
    
    if report["warnings"]:
        print("\n🚨 WARNINGS:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    
    if report["suggestions"]:
        print("\n💡 SUGGESTIONS:")
        for suggestion in report["suggestions"]:
            print(f"  - {suggestion}")
    
    print("\n📊 STATISTICS:")
    for cs_id, stats in report["stats"].items():
        print(f"\n  Condition Set: {cs_id}")
        print(f"    Total available conditions: {stats['total_available']}")
        print(f"    Would select: {stats['would_select']}")
        print(f"    Available types: {stats['types_available']}")
        if stats['type_families']:
            print("    Type families:")
            for base_type, variants in stats['type_families'].items():
                print(f"      {base_type}: {variants}")

if __name__ == "__main__":
    main()
