"""
Process ics files and verify uuid
Required modules: ics
"""
import uuid
import re
import typing as T
import ics


uuid_map: T.Dict[str, str] = dict()

uid_storage: T.Set[str] = set()

organizer_re: re.Pattern = re.compile(
    r"(^ORGANIZER;.*mailto.*$)", re.MULTILINE)
attendees_re: re.Pattern = re.compile(
    r"(^ATTENDEE;.*mailto.*$)", re.MULTILINE)
mailto_re: re.Pattern = re.compile(
    r"mailto:[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


def process_ics(cal: ics.Calendar) -> str:
    """
    Process the uploaded ics file
    """
    uuid_str: str = str(uuid.uuid4())
    # process the events
    events: T.List[ics.Event] = cal.events
    for event in events:
        store_success: bool = map_uuid_to_event(uuid_str, event)
        if not store_success:
            return None
    return uuid_str


def extract_organizer_and_attendees(event: ics.Event) -> T.Tuple[T.Set[str], T.Set[str]]:
    """
    Extract the organizer and attendee fields according to the event
    """

    event_string: str = str(event)
    organizer_match: T.List[str] = organizer_re.findall(event_string)
    attendees_match: T.List[str] = attendees_re.findall(event_string)

    def extract_email(line: str) -> str:
        """
        Extract the email address from the organizer or attendee line
        """
        mailto_match: re.Match = mailto_re.search(line)
        if not mailto_match:
            return None
        email_address: str = line[
            mailto_match.start()+len('mailto:'):mailto_match.end()]
        return email_address if email_address else None

    # do not filter out the None, we might need to emit a warning on those email addresses
    organizer_set: T.Set[str] = set(map(extract_email, organizer_match))
    attendees_set: T.Set[str] = set(map(extract_email, attendees_match))

    # organizers are filtered out from attendees
    attendees_set.difference_update(organizer_set)

    return (organizer_set, attendees_set)


def map_uuid_to_event(uuid_str: str, event: ics.Event) -> bool:
    """
    Map the uuid to the event
    TODO
    """
    try:
        uuid_map[uuid_str] = event.uid

        # (event.begin.datetime, event.end.datetime, event.description)
    except Exception as e:
        print(e)
        return False
    return True


def verify_uuid(uuid_str: str) -> bool:
    """
    Verify a uuid
    TODO
    """
    try:
        verification_success: bool = uuid_str in uuid_map
    except Exception as e:
        print(e)
        return False
    return verification_success
