import configparser

class TuningConfig:
    _file = "./src/model/tuning/config.ini"
    
    _config = configparser.ConfigParser()
    _config.read(_file)

    # Convert .ini sections back to dictionary
    _sweep_config = {"method": _config.get("sweep", "method", fallback="random")}
    _metric = {
        "name": _config.get("metric", "name", fallback="accuracy"),
        "goal": _config.get("metric", "goal", fallback="maximize")
    }

    # Parse parameters with multiple values as lists
    _parameters = {}
    if _config.has_section("parameters"):
        for key, value in _config.items("parameters"):
            values = [v.strip() for v in value.split(",")]
            if len(values) > 1:
                _parameters[key] = {"values": [float(v) if v.replace('.', '', 1).isdigit() else v for v in values]}
            else:
                single_value = values[0]
                _parameters[key] = {"value": float(single_value) if single_value.replace('.', '', 1).isdigit() else single_value}
    
    _sweep_config["metric"] = _metric
    _sweep_config["parameters"] = _parameters
    
    @staticmethod
    def get_config():
        return TuningConfig._sweep_config