import tqdm


ENTITY_REWRITE = {
    "'s": "",
    "’s": "",
    "'": "",
    "The ": "",
    "the ": "",
    "US": "United States",
    "U.S.": "United States",
    "EU": "European Union",
    "F.B.I": "FBI",
}


def normalize_entity(e):
    for k, v in ENTITY_REWRITE.items():
        e = e.replace(k, v)
    return e


def get_entities_from_doc(doc):
    # filter entity type
    # blocked_entity_type = {'DATE', 'TIME', 'CARDINAL', 'PERCENT', 'ORDINAL', 'MONEY', 'QUANTITY'}
    blocked_entity_type = {}
    entities = [e.text for e in doc.ents]
    return entities


def get_entity_types_from_doc(doc):
    # filter entity type
    # blocked_entity_type = {'DATE', 'TIME', 'CARDINAL', 'PERCENT', 'ORDINAL', 'MONEY', 'QUANTITY'}
    blocked_entity_type = {}
    entities = [e.label_ for e in doc.ents]
    return entities


def get_entities(nlp, texts, n_process=16):
    docs = nlp.pipe(texts, n_process=n_process, batch_size=64)
    # docs = list(docs)
    entities = []
    entity_types = []
    for doc in tqdm.tqdm(docs, total=len(texts)):
        entities.append(get_entities_from_doc(doc))
        entity_types.append(get_entity_types_from_doc(doc))
    # entities = [get_entities_from_doc(doc) for doc in docs]
    # entity_types = [get_entity_types_from_doc(doc) for doc in docs]
    return entities, entity_types


def evaluate_entity_precision_constraint(
    entity_lead, entity_type_lead, entity_maintext
):
    allowed_entities = [
        "PERCENT",
        "MONEY",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
        "DATE",
        "TIME",
        "PERSON",
        "ORG",
    ]
    entity_lead = [
        e for e, t in zip(entity_lead, entity_type_lead) if t in allowed_entities
    ]
    entity_lead = set(entity_lead)
    if len(entity_lead) == 0:
        return 1
    entity_maintext = set(entity_maintext)
    return len(entity_lead.intersection(entity_maintext)) / len(entity_lead)


def evaluate_entity_precision(entity_lead, entity_maintext):
    entity_lead = set(entity_lead)
    if len(entity_lead) == 0:
        return 1
    entity_maintext = set(entity_maintext)
    return len(entity_lead.intersection(entity_maintext)) / len(entity_lead)
