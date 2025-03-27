# TMfPT: master-project

>TMfPT = Text Mining for Patient Timeline

This repository contains the code for the master project that was conducted Spring 2025 as the concluding project for a Master’s in Computer Science with a specialization in Artificial Intelligence degree at the Norwegian University of Science and Technology (NTNU). It includes the proposed pipeline for using text mining to extract patient timelines from Norwegian patient journals. It specifically includes the text mining methods, but does not include the used dataset and trained models because of ethical issues.

## How to run
The following sections describe how to use the pipeline.
### Requirements
This project requires Python (tested on v.3.12.3) and Java (tested on v.21). Dependency management in Python is handled by Poetry.

```
git submodule init
poetry shell
poetry install
```

Also you need to install an external framework named HeidelTime [(Doc:How to install HeidelTime)](/src/textmining/heideltime/README.md)

### Usage
The following file is an example of how to run the pipeline.
```
poetry run python scripts/pipeline/main.py
```

## How to train
The following sections describe how the text mining models can be trained.

### Configuration

The configuration file for the pipeline contains the path for each of the text mining component configuration file. Each text mining component can be trained individually by changing the `load` parameter in the config files for the given text mining component to `true`. To allow tuning the hyperparameters in the `train.parameters`section, change the `tune`parameter to `true`.


### Dataset format
The project does not use a standardized format but uses two local type. The main one consists of the following columns: 

- ``Text``: The entity text found in the original text.
- ``Id``: The id used to connect the relations between the entities.
- `MedicalEntity`: The medical entity category that the entity is associated to.
- ``DCT``: The DocTime relation that the entity is associated to.
- `TIMEX`: The time expression category that the e`tity is associated to.
- ``Context``: The entity's context.

The other one holds the relations between the different entities:

- `FROM`: The from entity text found in the original text.
- `FROM_Id`: The from entity id.
- `FROM_CONTEXT`: The from entity's context.
- `TO`: The to entity text found in the original text
- `TO_Id`: The to entity id.
-`TO_CONTEXT`: The to entity context.
- `RELATION`: The associated relation between the two entities.

The file [convert_from_cas_json.py](scripts/util/convert_from_cas_json.py), converts from CAS JSON format to our local data format.