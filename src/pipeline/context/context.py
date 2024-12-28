import configparser
from pipeline.setup import Pipeline

if __name__ == "__main__":
    
    lengths = [32, 64, 128, 256, 512]
    
    config_file_path = './src/pipeline/bilstmcrf-window.ini'
    for length in lengths:
        print("CONTEXT LENGTH", length)
        config = configparser.ConfigParser()

        # Read the existing config file
        config.read(config_file_path)

        # Check if the 'DEFAULT' section or the relevant section exists
        if 'DEFAULT' in config:
            # Update the max_length value
            config['MODEL']['max_length'] = str(length)
            config['train.parameters']['window'] = str(length)
        else:
            raise ValueError("The specified section does not exist in the config file.")

        # Write the updated config back to the file
        with open(config_file_path, 'w') as configfile:
            config.write(configfile)
        pipeline = Pipeline(config_file_path, 'NorSynthClinical', align=False)

