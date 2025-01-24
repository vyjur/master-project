from structure.relation import Relation


def find_duplicates(rel_entities, across=False):
    ### Across: True, makes edges between them instead
    duplicates = [] 
    for i, ent_i in enumerate(rel_entities):
        for j in range(i + 1, len(rel_entities)):  # Avoid redundant comparisons
            ent_j = rel_entities[j]
            if ent_i.value == ent_j.value and ent_i.type == ent_j.type:                        
                # TODO: does this work
                ent_i.context += ent_j.context
    return duplicates


def remove_duplicates(rel_entities, duplicates):
    # Sort and remove duplicates in reverse to avoid index shift issues
    duplicates = list(set(duplicates))
    duplicates.sort(reverse=True)
    for index in duplicates:
        del rel_entities[index]  # Remove by index directly
    return rel_entities
