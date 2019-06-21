"""
Process ics files and verify uuid
"""
import uuid
import arrow
import ics

uuid_map = dict()


def process_ics(cal: ics.Calendar) -> str:
    """
    Process the uploaded ics file
    """
    uuid_str = str(uuid.uuid4())
    # process the events
    events = cal.events
    for event in events:
        store_success = map_uuid_to_event(uuid_str, event)
        if not store_success:
            return None
    return uuid_str


def map_uuid_to_event(uuid_str: str, event: arrow.Arrow) -> bool:
    """
    Map the uuid to the datetime
    """
    try:
        uuid_map[uuid_str] = (event.begin.datetime, event.end.datetime)
    except Exception as e:
        print(e)
        return False
    return True


def verify_uuid(uuid_str: str) -> bool:
    """
    Verify a uuid from the database
    """
    try:
        verification_success = uuid_str in uuid_map
    except Exception as e:
        print(e)
        return False
    return verification_success
