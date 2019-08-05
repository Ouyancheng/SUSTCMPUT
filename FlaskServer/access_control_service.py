"""
A module to connect with access control service
"""
import json
import typing as T
import requests
import arrow

access_control_service_address = \
    'http://162.246.156.210:5005/blockchain/'


def check_valid_entity(identity: str) -> bool:
    """
    Check the address
    """
    json_str: str = json.dumps({
        "address": identity
    })
    response: requests.Response = requests.get(access_control_service_address+'checkEntity',
                                               json=json_str)
    if response.status_code == 200:
        try:
            response_data: T.Dict[str, T.Any] = response.json()
            return response_data['data'] if response_data['data'] else False
        except Exception as e:
            print(e)
            return False
    else:
        return False


def add_new_entity(identity: str,
                   start: arrow.Arrow,
                   expiry: arrow.Arrow,
                   is_organizer: bool,
                   write_permission: bool) -> bool:
    """
    Add a new address
    Request format:
    {
        "address": "{identity}",
        "valid": "True",              // always
        "expiry": "",                 // "%d.%m.%Y %H:%M:%S"
        "start": "",                  // same as above
        "permission": 0,              // 0 for read and 1 for write
        "grantFlag": "True"           // visitor: "False", organizer: "True"
    }
    """
    json_str: str = json.dumps({
        'address': identity,
        'valid': 'True',
        'expiry': expiry.strftime("%d.%m.%Y %H:%M:%S"),
        'start': start.strftime("%d.%m.%Y %H:%M:%S"),
        'permission': (1 if write_permission else 0),
        'grantFlag': ('True' if is_organizer else 'False')
    })
    try:
        response: requests.Response = requests.post(
            access_control_service_address+'registerEntity', json=json_str)
        if response.status_code != 200:
            return False
    except Exception as e:
        print(e)
        return False
    return True

#pylint: disable-msg=too-many-arguments


def add_new_access(src_identity: str,
                   dst_identity: str,
                   uid: str,
                   start: arrow.Arrow,
                   expiry: arrow.Arrow,
                   resource: str,
                   imply: T.List[str],
                   excludes: T.List[str]) -> bool:
    """
    Add a new access to resource, which includes the 'includes' field
    Request format:
    {
        "src": "{src_identity}",          // organizer's email / identity
        "dst": "{dst_identity}",          // invitee's email / identity
        "valid": "True",                  // always
        "expiry": 0,                      // the one just set before
        "start": 0,                       // mentioned before
        "hash": "{uid}",                  // key for access rule, unique all the time
        "resource": "{resource}",         // destination meeting room
        "imply": ["resources"]            // array: path along the destination, order matters
        "excludes": ["resources"]         // array: don't want someone to do something, order doesn't matter
    }
    """
    json_str: str = json.dumps({
        "src": src_identity,
        "dst": dst_identity,
        "valid": "True",
        "start": start.strftime("%d.%m.%Y %H:%M:%S"),
        "expiry": expiry.strftime("%d.%m.%Y %H:%M:%S"),
        "hash": uid,
        "resource": resource,
        "imply": imply,
        "excludes": excludes
    })
    try:
        response: requests.Response = requests.post(
            access_control_service_address+'registerAccess', json=json_str)
        if response.status_code != 200:
            return False
    except Exception as e:
        print(e)
        return False
    return True


def check_accessible_resources(src_identity: str,
                               dst_identity: str,
                               uid: str,
                               resource: str) -> bool:
    """
    TODO
    Request:
    {
        "src":
        "dst":
        "hash": "DOT10"
        "resource": "{resource}"
    }
    Response:
    {
        "valid": "",
        "main": "{destination room}"
    }
    // ask Kalvin for unspecified resources......
    // ask him whether the resource is in an accessible room......
    """
    json_str: str = json.dumps({
        "src": src_identity,
        "dst": dst_identity,
        "hash": uid,
        "resource": resource
    })
    try:
        response: requests.Response = requests.post(
            access_control_service_address+'checkAccess', json=json_str)
        if response.status_code == 200:
            response_data: T.Dict[str, T.Any] = response.json()
            if not response_data["valid"]:
                return False
            if response_data["valid"] == "True":
                return True
            if response_data["valid"] == "False":
                return False
            # there is one case: unspecified
            if not response_data["main"]:
                return False
            # TODO
            return None
    except Exception as e:
        print(e)
        return False
