from structure.enum import TAGS
from dateutil.parser import parse


def convert_to_input(input_tag_type, e, single=True, start=True):
    single = True
    entity = e["MedicalEntity"] if "MedicalEntity" in e else None
    date = e["TIMEX"] if "TIMEX" in e else None

    match input_tag_type:
        case TAGS.SIMPLE:
            return str(e["Context"]).replace(str(e["Text"]), f"<TAG>{e['Text']}</TAG>")
        case TAGS.XML:
            cat = "TAG"
            if entity:
                cat = "EVENT"
            elif date:
                cat = "TIMEX"

            if start and not single:
                return str(e["Context"]).replace(
                    str(e["Text"]), f"<{cat}A>{e['Text']}</{cat}A>"
                )
            elif not start and not single:
                return str(e["Context"]).replace(
                    str(e["Text"]), f"<{cat}B>{e['Text']}</{cat}B>"
                )

            return str(e["Context"]).replace(
                str(e["Text"]), f"<{cat}>{e['Text']}</{cat}>"
            )
        case TAGS.SOURCE:
            if entity and single:
                return str(e["Context"]).replace(str(e["Text"]), f"es {e['Text']} ee")
            elif entity and not single:
                if start:
                    return str(e["Context"]).replace(
                        str(e["Text"]), f"eas {e['Text']} eae"
                    )
                else:
                    return str(e["Context"]).replace(
                        str(e["Text"]), f"ebs {e['Text']} ebe"
                    )
            elif date:
                return str(e["Context"]).replace(str(e["Text"]), f"ts {e['Text']}ts")
            else:
                return str(e["Context"])
        case TAGS.CUSTOM:
            if entity and single:
                return str(e["Context"]).replace(str(e["Text"]), f"es {e['Text']}  ee")
            elif entity and not single:
                if start:
                    return str(e["Context"]).replace(
                        str(e["Text"]), f"as es {e['Text']} ee ae"
                    )
                else:
                    return str(e["Context"]).replace(
                        str(e["Text"]), f"bs es {e['Text']} ee be"
                    )
            elif date:
                return str(e["Context"]).replace(str(e["Text"]), f"ts {e['Text']} ts")
            else:
                return str(e["Context"])
        case _:
            return str(e["Context"])


def is_date(text):
    try:
        parse(text)
        return True
    except ValueError:
        return False

