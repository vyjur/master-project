import configparser


class TuningConfig:
    _file = "./src/model/tuning/config.ini"

    _config = configparser.ConfigParser()
    _config.read(_file)

    # Convert .ini sections back to dictionary
    _sweep_config = {"method": _config.get("sweep", "method", fallback="random")}
    _metric = {
        "name": _config.get("metric", "name", fallback="accuracy"),
        "goal": _config.get("metric", "goal", fallback="maximize"),
    }

    # Parse parameters with multiple values as lists
    _parameters = {}
    if _config.has_section("parameters"):
        for key, value in _config.items("parameters"):
            values = [
                v.strip().lower() for v in value.split(",")
            ]  # Convert to lowercase for boolean handling

            # Check if all values are boolean
            if all(v in ["true", "false"] for v in values):
                converted_values = [
                    v == "true" for v in values
                ]  # Convert to actual boolean values
            else:
                converted_values = [
                    float(v) if v.replace(".", "", 1).isdigit() else v for v in values
                ]  # Convert numeric values

            if len(converted_values) > 1:
                _parameters[key] = {"values": converted_values}
            else:
                _parameters[key] = {"value": converted_values[0]}

    _sweep_config["metric"] = _metric
    _sweep_config["parameters"] = _parameters

    @staticmethod
    def get_config():
        return TuningConfig._sweep_config

